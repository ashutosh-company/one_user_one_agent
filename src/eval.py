"""
Outcome judgment and Causal Impact Measurement.
Evaluates goal success strictly within the post-decision window and measures
unconfounded Intent-to-Treat (ITT) impact against the randomized holdout.
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from src.data import get_data_pack, load_csv
from src.state import load_viewer_state, save_viewer_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RUN = PROJECT_ROOT / "sample_run"
OUTCOMES_PATH = SAMPLE_RUN / "outcomes.jsonl"
IMPACT_REPORT_PATH = SAMPLE_RUN / "impact_report.md"
FINAL_REPORT_PATH = SAMPLE_RUN / "final_report.md"


def judge_outcomes(sim_events_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Evaluates goal outcomes for all viewers who received an active goal."""
    OUTCOMES_PATH.write_text("", encoding="utf-8")
    pack = get_data_pack()

    # Load simulated outcome events if available
    sim_events = []
    if sim_events_path and sim_events_path.exists():
        with open(sim_events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        sim_events.append(json.loads(line))
                    except Exception:
                        pass

    # Index simulated events by user
    events_by_viewer: Dict[str, List[Dict[str, Any]]] = {}
    for ev in sim_events:
        uid = ev.get("user_id")
        if uid:
            events_by_viewer.setdefault(uid, []).append(ev)

    # Establish simulation evaluation horizon
    if sim_events:
        max_ts = max((e.get("received_at", "") for e in sim_events if e.get("received_at")), default="")
        try:
            eval_now = datetime.fromisoformat(max_ts.replace("Z", "+00:00")) + timedelta(hours=1)
        except Exception:
            eval_now = datetime(2026, 10, 16, 0, 0, 0, tzinfo=timezone.utc)
    else:
        eval_now = datetime(2026, 10, 16, 0, 0, 0, tzinfo=timezone.utc)

    # Success event mapping per goal
    success_event_map = {
        "G01": "play_started",
        "G02": "episode_completed",
        "G03": "play_started",
        "G04": "purchase_completed",
        "G05": "purchase_completed",
        "G06": "purchase_completed",
        "G07": "play_started",
        "G08": "play_started",
        "G09": "play_started",
        "G12": "play_started",
    }
    window_hours_map = {
        "G01": 24, "G02": 72, "G03": 48, "G04": 168, "G05": 168,
        "G06": 336, "G07": 72, "G08": 168, "G09": 336, "G12": 168,
    }

    outcomes = []
    state_dir = SAMPLE_RUN / "viewer_state"
    if not state_dir.exists():
        return outcomes

    for p in state_dir.glob("*.json"):
        vstate = load_viewer_state(p.stem)
        active_goal = vstate.get("active_goal")
        if not active_goal:
            continue
            
        decisions = vstate.get("decisions", [])
        if not decisions:
            continue
            
        decided_at_str = decisions[0].get("decided_at")
        if not decided_at_str:
            continue

        try:
            decided_at = datetime.fromisoformat(decided_at_str.replace("Z", "+00:00"))
        except Exception:
            decided_at = eval_now

        gid = active_goal.get("goal_id")
        gname = active_goal.get("goal_name", gid)
        target_event = success_event_map.get(gid)
        window_h = window_hours_map.get(gid, 72)
        deadline = decided_at + timedelta(hours=window_h)

        # Attribution Filter: only events strictly AFTER decision and BEFORE deadline
        user_events = events_by_viewer.get(p.stem, [])
        attributable = []
        for ev in user_events:
            if ev.get("event") == target_event:
                ev_time_str = ev.get("received_at", "")
                try:
                    ev_dt = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                    if decided_at < ev_dt <= deadline:
                        # Extra check for purchases (must be non-trial for G04/G05)
                        if gid in ("G04", "G05") and (ev.get("properties") or {}).get("is_trial", True):
                            continue
                        attributable.append(ev)
                except Exception:
                    pass

        if attributable:
            outcome_val = "SUCCEEDED"
            reason = f"Attributable event {attributable[0].get('event_id')} observed at {attributable[0].get('received_at')}"
            supporting_id = attributable[0].get("event_id")
        elif eval_now < deadline:
            outcome_val = "INCONCLUSIVE"
            reason = f"Goal measurement deadline {deadline.isoformat()} has not yet elapsed"
            supporting_id = None
        else:
            outcome_val = "FAILED"
            reason = f"No attributable {target_event} observed within {window_h}h measurement window"
            supporting_id = None

        record = {
            "viewer_id": p.stem,
            "goal_id": gid,
            "goal_name": gname,
            "decided_at": decided_at_str,
            "deadline": deadline.isoformat(),
            "outcome": outcome_val,
            "supporting_event_id": supporting_id,
            "reason": reason,
            "judged_at": eval_now.isoformat()
        }
        outcomes.append(record)

        with open(OUTCOMES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Feedback learning into durable viewer memory
        vstate.setdefault("outcomes", []).append(record)
        save_viewer_state(p.stem, vstate)

    return outcomes


def compute_impact_report(outcomes: List[Dict[str, Any]]):
    """Generates Intent-to-Treat causal impact report and explains selection bias."""
    arms_rows = load_csv("replay/arms.csv")
    arms = {r["user_id"]: r["arm"] for r in arms_rows}

    outcome_map = {o["viewer_id"]: o["outcome"] for o in outcomes}
    
    agent_viewers = [uid for uid, arm in arms.items() if arm == "agent"]
    holdout_viewers = [uid for uid, arm in arms.items() if arm == "holdout"]

    agent_success = sum(1 for uid in agent_viewers if outcome_map.get(uid) == "SUCCEEDED")
    holdout_success = sum(1 for uid in holdout_viewers if outcome_map.get(uid) == "SUCCEEDED")

    agent_n = len(agent_viewers) or 1
    holdout_n = len(holdout_viewers) or 1

    agent_rate = agent_success / agent_n
    holdout_rate = holdout_success / holdout_n
    itt_lift = (agent_rate - holdout_rate) * 100

    report = f"""# Causal Impact Report

## 1. Intent-to-Treat (ITT) Analysis vs. Randomized Holdout

The agent's true causal effect is measured by comparing all viewers randomly assigned
to the **Agent Arm** against all viewers in the **Holdout Arm** (unexposed counterfactual).

| Arm | Total Viewers | Successful Goals | Conversion Rate |
| :--- | :--- | :--- | :--- |
| **Agent (Treatment)** | {agent_n:,} | {agent_success:,} | **{agent_rate:.2%}** |
| **Holdout (Control)** | {holdout_n:,} | {holdout_success:,} | **{holdout_rate:.2%}** |

**Causal ITT Uplift**: **{itt_lift:+.2f} percentage points**

---

## 2. Why the Pilot Report's "Served vs. Unserved" Comparison Is Flawed

In the earlier pilot report (`pilot/pilot_summary_report.md`), the team claimed a +22.4% uplift
by comparing **"served viewers" against "all other users"**. This comparison is severely misleading due to:

1. **Selection Bias**: The agent only serves viewers who open the app and meet engagement eligibility.
   These viewers are inherently more motivated to watch content than dormant users who never visited.
2. **Activity Bias**: Users who open the app have non-zero conversion probability; users who never opened
   the app have zero probability. Grouping inactive users into the control artificially deflates the baseline.
3. **The Counterfactual Solution**: The randomized holdout arm provides the *only valid counterfactual*.
   Because holdout viewers also opened the app during the replay day under identical conditions but were
   intentionally withheld from intervention, the difference represents the true causal effect of the agent.

---

## 3. Outcome Summary
- **Total Judged Viewers**: {len(outcomes):,}
- **Succeeded**: {sum(1 for o in outcomes if o['outcome'] == 'SUCCEEDED'):,}
- **Failed**: {sum(1 for o in outcomes if o['outcome'] == 'FAILED'):,}
- **Inconclusive**: {sum(1 for o in outcomes if o['outcome'] == 'INCONCLUSIVE'):,}
"""
    with open(IMPACT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Final Submission Report\n\n{report}")

    print(f"Generated: {IMPACT_REPORT_PATH}")
    print(f"Generated: {FINAL_REPORT_PATH}")
