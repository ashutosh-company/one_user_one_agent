"""
Test suite for late and duplicate trigger handling in the LangGraph agent.
"""
from datetime import datetime, timezone, timedelta
from src.agent import get_viewer_agent
from src.state import load_viewer_state


def test_duplicate_trigger_is_idempotent():
    """Verifies that an identical trigger event_id is skipped and does not re-publish."""
    agent = get_viewer_agent()
    now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    trigger = {
        "event_id": "test-dup-101",
        "user_id": "U-TEST-DUP",
        "event": "app_open",
        "received_at": "2026-10-01T10:00:00Z",
        "deliver_at": "2026-10-01T10:00:00Z",
        "app_dialect": "d1",
        "properties": {}
    }

    # First execution
    state1 = {
        "viewer_id": "U-TEST-DUP",
        "trigger": trigger,
        "now": now,
        "arm": "agent",
        "is_duplicate": False,
        "is_stale": False,
    }
    res1 = agent.invoke(state1)
    assert res1["decision_record"]["status"] == "PUBLISHED"

    # Second execution (marked as duplicate because event_id is already in processed state)
    state2 = {
        "viewer_id": "U-TEST-DUP",
        "trigger": trigger,
        "now": now,
        "arm": "agent",
        "is_duplicate": True,
        "is_stale": False,
    }
    res2 = agent.invoke(state2)
    assert res2["decision_record"]["status"] == "SKIPPED"
    assert res2["decision_record"]["action"] == "NONE"


def test_stale_offline_buffered_trigger_abstains():
    """Verifies that offline-buffered triggers hours/days old result in NO_ACTION."""
    agent = get_viewer_agent()
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    stale_trigger = {
        "event_id": "test-stale-202",
        "user_id": "U-TEST-STALE",
        "event": "app_open",
        "received_at": "2026-09-28T12:00:00Z",  # 3 days old
        "deliver_at": "2026-10-01T12:00:00Z",
        "app_dialect": "d2",
        "properties": {}
    }

    state = {
        "viewer_id": "U-TEST-STALE",
        "trigger": stale_trigger,
        "now": now,
        "arm": "agent",
        "event_age_s": 259200,  # 3 days
        "is_duplicate": False,
        "is_stale": True,
    }
    result = agent.invoke(state)
    assert result["decision_record"]["action"] == "NONE"
    assert "old" in result["decision_record"]["reason_for_no_action"].lower()
