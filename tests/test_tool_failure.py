"""
Test suite for tool failures, holdout guardrail, and hallucination veto in LangGraph.
"""
from datetime import datetime, timezone
from src.agent import get_viewer_agent
from src.data import get_data_pack


def test_holdout_guardrail_blocks_action():
    """Verifies that viewers in the holdout arm receive NO action under any circumstances."""
    agent = get_viewer_agent()
    state = {
        "viewer_id": "U-TEST-HOLDOUT",
        "trigger": {
            "event_id": "test-holdout-505",
            "user_id": "U-TEST-HOLDOUT",
            "event": "app_open",
            "app_dialect": "d1",
            "deliver_at": "2026-10-01T10:00:00Z",
            "received_at": "2026-10-01T10:00:00Z"
        },
        "now": datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arm": "holdout"  # Randomized Holdout
    }
    result = agent.invoke(state)
    decision = result["decision_record"]
    assert decision["action"] == "NONE"
    assert "holdout" in decision["reason_for_no_action"].lower()
    assert "holdout" in decision["checks"]["failed"]


def test_llm_hallucinated_claim_is_vetoed():
    """Verifies that an unsupported model claim ('confirmed in billing') is caught and rejected."""
    pack = get_data_pack()
    eid = "fault-event-606"
    pack.model_faults[eid] = {
        "trigger_event_id": eid,
        "fault": "unsupported_claim",
        "model_output": {"goal_id": "G04", "eligible": True, "reason": "confirmed in billing"}
    }

    agent = get_viewer_agent()
    state = {
        "viewer_id": "U-TEST-FAULT",
        "trigger": {
            "event_id": eid,
            "user_id": "U-TEST-FAULT",
            "event": "app_open",
            "app_dialect": "d1",
            "deliver_at": "2026-10-01T10:00:00Z",
            "received_at": "2026-10-01T10:00:00Z"
        },
        "now": datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arm": "agent"
    }
    result = agent.invoke(state)
    decision = result["decision_record"]
    assert decision["action"] == "NONE"
    assert "llm_unsupported_claim" in decision["checks"]["failed"]
