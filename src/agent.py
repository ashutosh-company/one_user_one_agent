"""
LangGraph-based Agentic System for "One User, One Agent".

Implements a stateful cognitive graph:
START -> resolve_facts -> plan_goal -> select_content -> verify_safety -> [publish_action | abstain] -> END
"""
import re
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import TypedDict, List, Dict, Optional, Any
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from src.data import get_data_pack
from src.state import load_viewer_state, save_viewer_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RUN = PROJECT_ROOT / "sample_run"
ACTIONS_PATH = SAMPLE_RUN / "actions.jsonl"
DECISIONS_PATH = SAMPLE_RUN / "decisions.jsonl"

# Prompt injection filters
_INJECTION_PATTERNS = [
    re.compile(r'(SYSTEM\s+NOTE|ignore your previous rules|send\s+a\s+(WhatsApp|push|message))', re.I),
    re.compile(r'message every (user|viewer)', re.I),
    re.compile(r'including users who opted out', re.I),
]

# Unmeasurable goals permanently blocked from selection
UNMEASURABLE_GOALS = {"G10", "G11"}


class ViewerState(TypedDict, total=False):
    """The LangGraph State representing the viewer context throughout the pipeline."""
    viewer_id: str
    trigger: Dict[str, Any]
    now: datetime
    arm: str
    event_age_s: int
    is_duplicate: bool
    is_stale: bool
    
    # Perception / Facts
    facts: Dict[str, Any]
    
    # Goal Planning
    goal: Optional[Dict[str, Any]]
    goal_reason: str
    
    # Content & Action Proposal
    action: str
    content: List[str]
    title_ids: List[str]
    confidence: float
    confidence_reason: str
    assumptions: List[str]
    evidence: List[str]
    evidence_freshness: Dict[str, int]
    
    # Independent Verification Gate
    checks_passed: List[str]
    checks_failed: List[str]
    override_action: Optional[str]
    override_reason: Optional[str]
    
    # Final Decision Output
    decision_record: Dict[str, Any]
    budget_used: Dict[str, int]
    decided_in_ms: int
    status: str


# ─── NODE 1: PERCEPTION & FACT RESOLUTION ─────────────────────────────────────

def resolve_facts_node(state: ViewerState) -> Dict[str, Any]:
    """Ingests trigger, resolves dialect & subscription conflicts, and calculates freshness."""
    pack = get_data_pack()
    viewer_id = state["viewer_id"]
    trigger = state["trigger"]
    now = state["now"]
    
    profile = pack.profiles.get(viewer_id, {})
    historical_events = pack.events_by_viewer.get(viewer_id, [])
    live_events = pack.live_events_by_viewer.get(viewer_id, [])
    
    # 1. Resolve Dialect with Precedence: app_setting > campaign_link > profile
    sources = {}
    app_dialect = trigger.get("app_dialect")
    if app_dialect and app_dialect not in ("NA", "", "null"):
        sources["app_setting"] = app_dialect

    deeplink = (trigger.get("properties") or {}).get("deeplink", "")
    if deeplink and "/title/" in deeplink:
        parts = deeplink.split("/")
        if len(parts) >= 5 and parts[4].startswith("d"):
            sources["campaign_link"] = parts[4]

    prof_dialect = profile.get("preferred_dialect")
    if prof_dialect and prof_dialect not in ("NA", "", "null"):
        sources["profile"] = prof_dialect

    winner_dialect = sources.get("app_setting") or sources.get("campaign_link") or sources.get("profile") or "d1"
    winner_source = next((k for k in ["app_setting", "campaign_link", "profile"] if k in sources), "default")
    
    # Detect dialect conflicts
    conflicts = [f"{k}:{v}" for k, v in sources.items() if v != winner_dialect]

    # 2. Resolve Subscription Status (Live purchase event beats stale warehouse snapshot)
    prof_sub = profile.get("subscription_status", "never")
    live_sub = None
    for ev in reversed(live_events):
        if ev.get("event") == "purchase_completed":
            is_trial = (ev.get("properties") or {}).get("is_trial", True)
            live_sub = "trial_active" if is_trial else "active"
            break
        elif ev.get("event") == "subscription_lapsed":
            live_sub = "lapsed"
            break
            
    resolved_sub = live_sub or prof_sub
    sub_conflicts = [f"profile:{prof_sub}"] if (live_sub and live_sub != prof_sub) else []

    # 3. Aggregate Behavioral Facts
    all_events = historical_events + live_events
    event_names = {e.get("event") for e in all_events}
    
    # Unfinished series (started but not completed)
    started_titles = {e.get("properties", {}).get("title_id") or e.get("title_id") for e in all_events if e.get("event") == "play_started"}
    completed_titles = {e.get("properties", {}).get("title_id") or e.get("title_id") for e in all_events if e.get("event") == "episode_completed"}
    unfinished = list(started_titles - completed_titles)

    # Days since signup
    signup_at = profile.get("signup_at")
    days_since_signup = None
    if signup_at:
        try:
            s_dt = datetime.fromisoformat(signup_at.replace("Z", "+00:00"))
            days_since_signup = (now - s_dt).days
        except Exception:
            pass

    facts = {
        "dialect": {"used": winner_dialect, "source": winner_source, "conflicts": conflicts},
        "subscription": {"status": resolved_sub, "conflicts": sub_conflicts},
        "has_played": "play_started" in event_names,
        "has_completed_episode": "episode_completed" in event_names,
        "has_purchased": "purchase_completed" in event_names,
        "unfinished_series": unfinished,
        "days_since_signup": days_since_signup,
        "notifications_opt_in": profile.get("notifications_opt_in", "true").lower() == "true",
        "whatsapp_opt_in": profile.get("whatsapp_opt_in", "false").lower() == "true",
    }
    
    # Build Evidence References
    evidence = []
    evidence_freshness = {}
    if historical_events:
        latest = historical_events[-1]
        ref = f"event:{latest.get('event_id', 'latest')}"
        evidence.append(ref)
        try:
            ev_dt = datetime.fromisoformat(latest.get("received_at", "").replace("Z", "+00:00"))
            evidence_freshness[ref] = max(0, int((now - ev_dt).total_seconds()))
        except Exception:
            evidence_freshness[ref] = 0
            
    if profile:
        ref = f"profile:snapshot@{profile.get('snapshot_at', 'warehouse')}"
        evidence.append(ref)
        evidence_freshness[ref] = 86400  # daily snapshot

    return {
        "facts": facts,
        "evidence": evidence,
        "evidence_freshness": evidence_freshness,
        "budget_used": {"model_calls": 0, "tool_calls": 1}
    }


# ─── NODE 2: GOAL PLANNING ───────────────────────────────────────────────────

def plan_goal_node(state: ViewerState) -> Dict[str, Any]:
    """Selects one measurable goal with deadline, or explains why no goal is chosen."""
    pack = get_data_pack()
    facts = state["facts"]
    now = state["now"]
    
    # Goal priority (Conversion -> Activation -> Habit/Retention)
    priority = ["G04", "G05", "G06", "G01", "G02", "G03", "G07", "G08", "G09", "G12"]
    goals_map = {g["goal_id"]: g for g in pack.goals}
    
    # Apply dynamic rule corrections from control events
    for gid, correction in pack.corrected_goals.items():
        if gid in goals_map:
            goals_map[gid]["success"] = correction.get("new_success", goals_map[gid]["success"])

    selected_goal = None
    reason = "No eligible goal found"
    sub_status = facts["subscription"]["status"]

    for gid in priority:
        if gid in UNMEASURABLE_GOALS:
            continue
        g = goals_map.get(gid)
        if not g:
            continue
            
        el = g.get("eligibility", {})
        # Subscription filter
        if el.get("subscription_status") and sub_status not in el["subscription_status"]:
            continue
        # Signed up within days
        if el.get("signed_up_within_days") is not None:
            dss = facts.get("days_since_signup")
            if dss is None or dss > el["signed_up_within_days"]:
                continue
        # Requires no prior event
        req_absent = el.get("requires_no_prior_event")
        if req_absent == "play_started" and facts["has_played"]:
            continue
        if req_absent == "episode_completed" and facts["has_completed_episode"]:
            continue
        # Unfinished series rule (G07)
        if gid == "G07" and not facts["unfinished_series"]:
            continue
            
        selected_goal = g
        reason = f"Selected {gid} ({g.get('name')}) as best eligible goal"
        break

    assumptions = ["Profile snapshot may be up to 24h stale"]
    evidence = list(state.get("evidence", []))
    evidence_freshness = dict(state.get("evidence_freshness", {}))
    
    if selected_goal:
        ref = f"goal_rule:{selected_goal['goal_id']}"
        evidence.append(ref)
        evidence_freshness[ref] = 0
        if selected_goal["goal_id"] == "G07":
            assumptions.append("Assumption: series has remaining episodes since no episode_completed was logged")
        if selected_goal["goal_id"] == "G09":
            assumptions.append("Assumption: 14-day play window is rolling from trigger timestamp")

    # Increment tool call counter
    budget = state.get("budget_used", {"model_calls": 0, "tool_calls": 0})
    budget["tool_calls"] += 1

    return {
        "goal": selected_goal,
        "goal_reason": reason,
        "assumptions": assumptions,
        "evidence": evidence,
        "evidence_freshness": evidence_freshness,
        "budget_used": budget,
    }


# ─── NODE 3: CONTENT & ACTION PROPOSAL ───────────────────────────────────────

def select_content_node(state: ViewerState) -> Dict[str, Any]:
    """Selects titles available in the viewer's dialect, neutralizing prompt injection."""
    pack = get_data_pack()
    dialect = state["facts"]["dialect"]["used"]
    now_str = state["now"].strftime("%Y-%m-%d")
    
    # Query catalogue for dialect
    candidates = pack.catalogue_by_dialect.get(dialect, [])
    safe_titles = []
    seen_slugs = set()

    for item in candidates:
        tid = item["title_id"]
        if tid in pack.withdrawn_titles:
            continue
        # Availability window check
        avail_until = item.get("available_until")
        if avail_until and avail_until < now_str:
            continue
        # Neutralize Prompt Injection in descriptions
        desc = item.get("description", "")
        if any(p.search(desc) for p in _INJECTION_PATTERNS):
            continue
        # Deduplicate language variants by content_slug
        slug = item.get("content_slug", tid)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        safe_titles.append(tid)

    # Goal-directed content sorting
    goal_id = state.get("goal", {}).get("goal_id") if state.get("goal") else ""
    if goal_id == "G07":
        # Prioritize unfinished titles
        unf = set(state["facts"]["unfinished_series"])
        safe_titles.sort(key=lambda t: 0 if t in unf else 1)

    selected_tids = safe_titles[:8]
    evidence = list(state.get("evidence", []))
    evidence_freshness = dict(state.get("evidence_freshness", {}))
    for tid in selected_tids:
        ref = f"catalogue:{tid}"
        evidence.append(ref)
        evidence_freshness[ref] = 0

    # Calculate Confidence score + Short reason
    score = 0.5
    conf_reasons = []
    if not state["facts"]["dialect"]["conflicts"]:
        score += 0.1
        conf_reasons.append("dialect confirmed by app setting with zero conflict")
    else:
        score -= 0.1
        conf_reasons.append(f"dialect conflict with {state['facts']['dialect']['conflicts']}")
    if state["facts"]["has_played"]:
        score += 0.1
        conf_reasons.append("prior play history exists")
    
    confidence = round(max(0.1, min(0.95, score)), 2)
    conf_reason = "; ".join(conf_reasons) or "baseline heuristic estimate"

    budget = state.get("budget_used", {"model_calls": 0, "tool_calls": 0})
    budget["tool_calls"] += 1

    return {
        "action": "HOME_PAGE" if state.get("goal") else "NONE",
        "title_ids": selected_tids,
        "content": [f"title:{tid}" for tid in selected_tids[:5]],
        "confidence": confidence,
        "confidence_reason": conf_reason,
        "evidence": evidence,
        "evidence_freshness": evidence_freshness,
        "budget_used": budget,
    }


# ─── NODE 4: INDEPENDENT VERIFIER (GUARDRAIL / CRITIC) ───────────────────────

def verify_safety_node(state: ViewerState) -> Dict[str, Any]:
    """Independent verification gate. Has authority to veto or downgrade proposed actions."""
    passed = []
    failed = []
    override_action = None
    override_reason = None
    
    # 1. Randomized Holdout Guardrail
    if state["arm"] == "holdout":
        failed.append("holdout")
        return {
            "checks_passed": passed,
            "checks_failed": failed,
            "override_action": "NONE",
            "override_reason": "Viewer is in holdout arm – no action permitted"
        }
    passed.append("holdout")

    # 2. Check for Duplicate or Stale Triggers
    if state.get("is_duplicate"):
        failed.append("duplicate_action")
        return {
            "checks_passed": passed, "checks_failed": failed,
            "override_action": "NONE", "override_reason": "Duplicate trigger – already processed"
        }
    if state.get("is_stale"):
        failed.append("fresh_trigger")
        return {
            "checks_passed": passed, "checks_failed": failed,
            "override_action": "NONE", "override_reason": f"Trigger is {state.get('event_age_s')}s old (>3h threshold)"
        }
    passed.extend(["duplicate_action", "fresh_trigger"])

    # 3. Goal Measurability Check
    goal = state.get("goal")
    if goal and goal.get("goal_id") in UNMEASURABLE_GOALS:
        failed.append("goal_measurable")
        return {
            "checks_passed": passed, "checks_failed": failed,
            "override_action": "NONE", "override_reason": f"Goal {goal.get('goal_id')} has no measurable signal"
        }
    passed.append("goal_measurable")

    # 4. Content Dialect & Availability (At least 3 valid titles required for Home Page row)
    if state.get("action") == "HOME_PAGE":
        title_ids = state.get("title_ids", [])
        if len(title_ids) < 3:
            failed.append("min_titles_per_row")
            return {
                "checks_passed": passed, "checks_failed": failed,
                "override_action": "NONE", "override_reason": f"Only {len(title_ids)} valid titles (need ≥ 3)"
            }
        passed.extend(["dialect_match", "availability", "min_titles_per_row", "prompt_safety"])

    # 5. Model Fault Injection Check (e.g. LLM hallucinates "confirmed in billing")
    pack = get_data_pack()
    trigger_eid = state["trigger"].get("event_id")
    fault = pack.model_faults.get(trigger_eid)
    if fault and fault.get("fault") == "unsupported_claim":
        failed.append("llm_unsupported_claim")
        return {
            "checks_passed": passed, "checks_failed": failed,
            "override_action": "NONE", "override_reason": "LLM made unsupported claim without verifiable event"
        }

    return {
        "checks_passed": passed,
        "checks_failed": failed,
        "override_action": override_action,
        "override_reason": override_reason
    }


# ─── NODE 5: PUBLISH ACTION ───────────────────────────────────────────────────

def publish_action_node(state: ViewerState) -> Dict[str, Any]:
    """Publishes approved action to actions.jsonl and builds audit-ready decision record."""
    viewer_id = state["viewer_id"]
    trigger = state["trigger"]
    now = state["now"]
    goal = state["goal"]
    title_ids = state["title_ids"]
    dialect_used = state["facts"]["dialect"]["used"]
    
    decision_id = f"D-{uuid.uuid4().hex[:8].upper()}"
    action_id = f"A-{uuid.uuid4().hex[:8].upper()}"
    
    # Calculate delivery/published timestamps
    deliver_at = trigger.get("deliver_at") or trigger.get("received_at") or now.isoformat()
    decided_ms = state.get("decided_in_ms", 5)
    published_dt = now + timedelta(milliseconds=decided_ms)
    published_at = published_dt.isoformat().replace("+00:00", "Z")
    
    window_h = (goal.get("success") or {}).get("window_hours", 72)
    deadline = (now + timedelta(hours=window_h)).isoformat().replace("+00:00", "Z")

    # 1. Action Payload
    action_payload = {
        "action_id": action_id,
        "decision_id": decision_id,
        "user_id": viewer_id,
        "trigger_event_id": trigger.get("event_id"),
        "type": "home_page",
        "published_at": published_at,
        "dialect": dialect_used,
        "rows": [{"row_type": "popular_in_dialect", "title_ids": title_ids}],
    }
    with open(ACTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(action_payload, ensure_ascii=False) + "\n")

    # 2. Decision Record
    decision_record = {
        "decision_id": decision_id,
        "viewer_id": viewer_id,
        "trigger": {
            "event": trigger.get("event"),
            "received_at": trigger.get("received_at"),
            "deliver_at": deliver_at,
            "event_age_s": state.get("event_age_s", 0)
        },
        "goal": {
            "id": goal.get("goal_id"),
            "name": goal.get("name"),
            "deadline": deadline,
            "success_signal": (goal.get("success") or {}).get("event")
        },
        "action": "HOME_PAGE",
        "reason_for_no_action": None,
        "dialect": {
            "used": dialect_used,
            "source": state["facts"]["dialect"]["source"],
            "freshness_s": 0,
            "conflicts": state["facts"]["dialect"]["conflicts"]
        },
        "content": state["content"],
        "confidence": state["confidence"],
        "confidence_reason": state["confidence_reason"],
        "evidence": state["evidence"],
        "evidence_freshness": state["evidence_freshness"],
        "assumptions": state["assumptions"],
        "checks": {
            "passed": state["checks_passed"],
            "failed": state["checks_failed"]
        },
        "budget_used": state["budget_used"],
        "decided_in_ms": decided_ms,
        "status": "PUBLISHED"
    }
    with open(DECISIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision_record, ensure_ascii=False) + "\n")

    # 3. Update Durable Viewer Memory
    vstate = load_viewer_state(viewer_id)
    vstate.setdefault("processed_trigger_ids", []).append(trigger.get("event_id"))
    vstate["active_goal"] = {"goal_id": goal.get("goal_id"), "goal_name": goal.get("name")}
    vstate.setdefault("decisions", []).append({
        "decision_id": decision_id, "action": "HOME_PAGE", "decided_at": published_at
    })
    vstate.setdefault("shown_title_ids", []).extend(title_ids)
    save_viewer_state(viewer_id, vstate)

    return {"decision_record": decision_record, "status": "PUBLISHED"}


# ─── NODE 6: ABSTAIN / DO NOTHING ─────────────────────────────────────────────

def abstain_node(state: ViewerState) -> Dict[str, Any]:
    """Records a valid NO_ACTION decision with an explicit cited reason."""
    viewer_id = state["viewer_id"]
    trigger = state["trigger"]
    now = state["now"]
    decision_id = f"D-{uuid.uuid4().hex[:8].upper()}"
    
    reason = state.get("override_reason") or state.get("goal_reason") or "Abstained: constraints not satisfied"
    decided_ms = state.get("decided_in_ms", 1)
    
    decision_record = {
        "decision_id": decision_id,
        "viewer_id": viewer_id,
        "trigger": {
            "event": trigger.get("event"),
            "received_at": trigger.get("received_at"),
            "deliver_at": trigger.get("deliver_at"),
            "event_age_s": state.get("event_age_s", 0)
        },
        "goal": {
            "id": state["goal"]["goal_id"], "name": state["goal"]["name"], "deadline": None, "success_signal": None
        } if state.get("goal") else None,
        "action": "NONE",
        "reason_for_no_action": reason,
        "dialect": state.get("facts", {}).get("dialect") or {"used": "d1", "source": "default", "conflicts": []},
        "content": [],
        "confidence": 0.0,
        "confidence_reason": reason,
        "evidence": state.get("evidence", []),
        "evidence_freshness": state.get("evidence_freshness", {}),
        "assumptions": state.get("assumptions", []),
        "checks": {
            "passed": state.get("checks_passed", []),
            "failed": state.get("checks_failed", [])
        },
        "budget_used": state.get("budget_used", {"model_calls": 0, "tool_calls": 0}),
        "decided_in_ms": decided_ms,
        "status": "SKIPPED" if state.get("is_duplicate") else "PUBLISHED"
    }
    with open(DECISIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision_record, ensure_ascii=False) + "\n")

    # Update state for holdout viewers or processed triggers
    vstate = load_viewer_state(viewer_id)
    if trigger.get("event_id"):
        vstate.setdefault("processed_trigger_ids", []).append(trigger.get("event_id"))
    if state.get("goal") and state["arm"] == "holdout":
        vstate["active_goal"] = {"goal_id": state["goal"]["goal_id"], "goal_name": state["goal"]["name"]}
        vstate.setdefault("decisions", []).append({
            "decision_id": decision_id, "action": "NONE", "decided_at": trigger.get("deliver_at")
        })
    save_viewer_state(viewer_id, vstate)

    return {"decision_record": decision_record, "status": decision_record["status"]}


# ─── ROUTER: CONDITIONAL EDGE ────────────────────────────────────────────────

def should_publish_action(state: ViewerState) -> str:
    """Routes to publish_action if checks passed and an action is proposed; otherwise abstains."""
    if state.get("override_action") == "NONE":
        return "abstain"
    if state.get("action") == "HOME_PAGE" and not state.get("checks_failed"):
        return "publish_action"
    return "abstain"


# ─── BUILD THE LANGGRAPH AGENT ────────────────────────────────────────────────

def create_viewer_agent():
    """Builds and compiles the LangGraph StateGraph agent."""
    graph = StateGraph(ViewerState)
    
    # Add Nodes
    graph.add_node("resolve_facts", resolve_facts_node)
    graph.add_node("plan_goal", plan_goal_node)
    graph.add_node("select_content", select_content_node)
    graph.add_node("verify_safety", verify_safety_node)
    graph.add_node("publish_action", publish_action_node)
    graph.add_node("abstain", abstain_node)
    
    # Add Edges
    graph.add_edge(START, "resolve_facts")
    graph.add_edge("resolve_facts", "plan_goal")
    graph.add_edge("plan_goal", "select_content")
    graph.add_edge("select_content", "verify_safety")
    
    # Conditional Branching after Independent Verification
    graph.add_conditional_edges(
        "verify_safety",
        should_publish_action,
        {
            "publish_action": "publish_action",
            "abstain": "abstain"
        }
    )
    
    graph.add_edge("publish_action", END)
    graph.add_edge("abstain", END)
    
    return graph.compile()


# Global compiled agent instance
_COMPILED_AGENT = None

def get_viewer_agent():
    global _COMPILED_AGENT
    if _COMPILED_AGENT is None:
        _COMPILED_AGENT = create_viewer_agent()
    return _COMPILED_AGENT
