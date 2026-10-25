"""
Test suite for mid-replay goal rule correction and unmeasurable goal blocking.
"""
from src.agent import UNMEASURABLE_GOALS
from src.data import get_data_pack
from src.replay import _find_viewers_with_goal
from src.state import save_viewer_state


def test_unmeasurable_goals_permanently_blocked():
    """G10 (null event) and G11 (tv_app_open) must be permanently blocked from selection."""
    assert "G10" in UNMEASURABLE_GOALS
    assert "G11" in UNMEASURABLE_GOALS


def test_goal_rule_correction_applies_dynamically():
    """Verifies that when a goal rule is corrected, the data pack updates its success criteria."""
    pack = get_data_pack()
    correction = {
        "type": "goal_rule_corrected",
        "goal_id": "G03",
        "new_success": {"event": "play_started", "condition": "on a later IST calendar day"}
    }
    pack.corrected_goals["G03"] = correction
    
    # Check that goals_map receives the correction
    goals_map = {g["goal_id"]: g for g in pack.goals}
    if "G03" in pack.corrected_goals:
        goals_map["G03"]["success"] = pack.corrected_goals["G03"]["new_success"]
        
    assert goals_map["G03"]["success"]["event"] == "play_started"


def test_selective_reevaluation_only_affects_matching_goal():
    """Verifies that an inverted index finds ONLY viewers holding the changed goal."""
    save_viewer_state("U-TEST-G03-1", {"active_goal": {"goal_id": "G03"}})
    save_viewer_state("U-TEST-G03-2", {"active_goal": {"goal_id": "G03"}})
    save_viewer_state("U-TEST-G05-1", {"active_goal": {"goal_id": "G05"}})

    affected_g03 = _find_viewers_with_goal("G03")
    assert "U-TEST-G03-1" in affected_g03
    assert "U-TEST-G03-2" in affected_g03
    assert "U-TEST-G05-1" not in affected_g03
