"""
Replay runner for the One User, One Agent assignment.
Drives triggers and control events through the compiled LangGraph agent.
"""
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from src.data import get_data_pack
from src.agent import get_viewer_agent
from src.state import load_viewer_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RUN = PROJECT_ROOT / "sample_run"
DECISIONS_PATH = SAMPLE_RUN / "decisions.jsonl"
ACTIONS_PATH = SAMPLE_RUN / "actions.jsonl"
LATENCY_PATH = SAMPLE_RUN / "latency_report.json"

STALE_TRIGGER_THRESHOLD_S = 3 * 3600  # 3 hours


def run_replay():
    """Runs the full replay stream through the LangGraph agent."""
    SAMPLE_RUN.mkdir(parents=True, exist_ok=True)
    DECISIONS_PATH.write_text("", encoding="utf-8")
    ACTIONS_PATH.write_text("", encoding="utf-8")

    pack = get_data_pack()
    agent = get_viewer_agent()

    # Merge triggers and control events in chronological order
    stream = []
    for t in pack.triggers:
        stream.append(("trigger", t.get("deliver_at", ""), t))
    for c in pack.control_events:
        stream.append(("control", c.get("at", ""), c))
    stream.sort(key=lambda x: x[1])

    total_triggers = sum(1 for s in stream if s[0] == "trigger")
    processed = 0
    latency_records = []

    print(f"Starting LangGraph Replay: {total_triggers} triggers + {len(pack.control_events)} control events")

    for kind, ts, item in stream:
        if kind == "control":
            ctype = item.get("type")
            print(f"  [CONTROL EVENT] {ctype} @ {ts}")
            if ctype == "title_withdrawn":
                pack.withdrawn_titles.add(item.get("title_id"))
            elif ctype == "goal_rule_corrected":
                gid = item.get("goal_id")
                pack.corrected_goals[gid] = item
                # Selective re-evaluation: re-evaluate only viewers with this active goal
                affected = _find_viewers_with_goal(gid)
                print(f"    -> Selectively re-evaluating {len(affected)} viewers holding {gid}")
                for uid in affected:
                    fake_trigger = {
                        "event_id": f"reeval-{uid}-{gid}",
                        "user_id": uid,
                        "event": "goal_rule_corrected",
                        "received_at": ts,
                        "deliver_at": ts,
                        "properties": {}
                    }
                    _process_trigger(agent, pack, fake_trigger, latency_records)
            continue

        # Process standard trigger through LangGraph
        _process_trigger(agent, pack, item, latency_records)
        processed += 1
        if processed % 500 == 0 or processed == total_triggers:
            print(f"  {processed}/{total_triggers} triggers processed by LangGraph...")

    # Write Latency Metrics Report
    durations = [r["decided_in_ms"] for r in latency_records]
    durations.sort()
    p50 = durations[int(len(durations) * 0.5)] if durations else 0
    p99 = durations[int(len(durations) * 0.99)] if durations else 0
    
    latency_report = {
        "total_triggers": total_triggers,
        "processed": processed,
        "p50_ms": p50,
        "p99_ms": p99,
        "max_ms": max(durations, default=0),
        "over_60s": sum(1 for d in durations if d > 60_000),
        "over_5min": sum(1 for d in durations if d > 300_000),
    }
    with open(LATENCY_PATH, "w", encoding="utf-8") as f:
        json.dump(latency_report, f, indent=2)

    print("\nReplay Complete!")
    print(f"  Decisions: {DECISIONS_PATH}")
    print(f"  Actions:   {ACTIONS_PATH}")
    print(f"  Latency:   p50={p50}ms, p99={p99}ms, max={max(durations, default=0)}ms")


def _process_trigger(agent, pack, trigger: dict, latency_records: list):
    """Feeds one trigger event through the LangGraph StateGraph."""
    t0 = time.time()
    uid = trigger.get("user_id")
    eid = trigger.get("event_id")
    deliver_at = trigger.get("deliver_at") or trigger.get("received_at", "")

    # Parse reference time
    try:
        now = datetime.fromisoformat(deliver_at.replace("Z", "+00:00"))
    except Exception:
        now = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Check for Stale / Offline-buffered trigger
    recv_str = trigger.get("received_at", deliver_at)
    try:
        recv_dt = datetime.fromisoformat(recv_str.replace("Z", "+00:00"))
        age_s = max(0, int((now - recv_dt).total_seconds()))
    except Exception:
        age_s = 0
    is_stale = (age_s > STALE_TRIGGER_THRESHOLD_S)

    # Check for Duplicate trigger
    vstate = load_viewer_state(uid)
    is_duplicate = eid in vstate.get("processed_trigger_ids", [])

    # Prepare initial LangGraph State
    initial_state = {
        "viewer_id": uid,
        "trigger": trigger,
        "now": now,
        "arm": pack.arms.get(uid, "agent"),
        "event_age_s": age_s,
        "is_duplicate": is_duplicate,
        "is_stale": is_stale,
    }

    # Execute the LangGraph Agent Workflow
    result = agent.invoke(initial_state)

    elapsed_ms = int((time.time() - t0) * 1000)
    decision = result.get("decision_record", {})
    latency_records.append({
        "decision_id": decision.get("decision_id"),
        "viewer_id": uid,
        "deliver_at": deliver_at,
        "decided_in_ms": elapsed_ms,
        "status": decision.get("status")
    })


def _find_viewers_with_goal(goal_id: str) -> list:
    """Inverted index lookup: returns viewers holding the specified active goal."""
    affected = []
    state_dir = SAMPLE_RUN / "viewer_state"
    if not state_dir.exists():
        return affected
    for p in state_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (data.get("active_goal") or {}).get("goal_id") == goal_id:
                affected.append(data.get("viewer_id", p.stem))
        except Exception:
            pass
    return affected
