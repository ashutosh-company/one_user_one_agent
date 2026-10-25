"""
Test suite for conflicting source resolution in the LangGraph agent.
"""
from datetime import datetime, timezone
from src.agent import get_viewer_agent
from src.data import get_data_pack


def test_three_way_dialect_conflict_precedence():
    """Verifies that app_setting wins over campaign_link and profile snapshot, logging conflicts."""
    agent = get_viewer_agent()
    pack = get_data_pack()
    
    # Inject profile with dialect d1
    pack.profiles["U-TEST-CONFLICT"] = {
        "user_id": "U-TEST-CONFLICT",
        "preferred_dialect": "d1",
        "snapshot_at": "2026-09-30T00:00:00Z",
        "subscription_status": "never",
    }

    trigger = {
        "event_id": "test-conflict-303",
        "user_id": "U-TEST-CONFLICT",
        "event": "app_open",
        "deliver_at": "2026-10-01T10:00:00Z",
        "received_at": "2026-10-01T10:00:00Z",
        "app_dialect": "d3",  # App setting is d3 (should win)
        "properties": {
            "deeplink": "ott://c/campaign01/d5/title/T0100"  # Campaign link is d5
        }
    }

    state = {
        "viewer_id": "U-TEST-CONFLICT",
        "trigger": trigger,
        "now": datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arm": "agent",
    }
    result = agent.invoke(state)
    decision = result["decision_record"]
    
    # app_setting d3 must win
    assert decision["dialect"]["used"] == "d3"
    assert decision["dialect"]["source"] == "app_setting"
    # Conflicts must be explicitly recorded
    conflicts = decision["dialect"]["conflicts"]
    assert any("profile:d1" in c for c in conflicts)
    assert any("campaign_link:d5" in c for c in conflicts)


def test_live_purchase_overrides_stale_profile():
    """Verifies that a live purchase event overrides a stale 'never' status in the profile."""
    pack = get_data_pack()
    pack.profiles["U-TEST-PURCHASE"] = {
        "user_id": "U-TEST-PURCHASE",
        "preferred_dialect": "d2",
        "subscription_status": "never",
        "snapshot_at": "2026-09-30T00:00:00Z",
    }
    pack.live_events_by_viewer["U-TEST-PURCHASE"] = [
        {
            "event": "purchase_completed",
            "received_at": "2026-10-01T09:58:00Z",
            "properties": {"is_trial": False, "plan_id": "P_MONTH"}
        }
    ]

    agent = get_viewer_agent()
    state = {
        "viewer_id": "U-TEST-PURCHASE",
        "trigger": {
            "event_id": "test-sub-404",
            "user_id": "U-TEST-PURCHASE",
            "event": "app_open",
            "app_dialect": "d2",
            "deliver_at": "2026-10-01T10:00:00Z",
            "received_at": "2026-10-01T10:00:00Z"
        },
        "now": datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arm": "agent"
    }
    result = agent.invoke(state)
    assert result["facts"]["subscription"]["status"] == "active"
    assert "profile:never" in result["facts"]["subscription"]["conflicts"]
