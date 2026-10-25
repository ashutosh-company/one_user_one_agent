"""
Durable per-viewer state management.
Each viewer has one dedicated agent owning their state in `sample_run/viewer_state/<viewer_id>.json`.
"""
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "sample_run" / "viewer_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def get_state_path(viewer_id: str) -> Path:
    return STATE_DIR / f"{viewer_id}.json"


def load_viewer_state(viewer_id: str) -> Dict[str, Any]:
    """Load durable state for a viewer, or initialize clean state."""
    path = get_state_path(viewer_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "viewer_id": viewer_id,
        "facts": {},
        "active_goal": None,
        "decisions": [],
        "outcomes": [],
        "processed_trigger_ids": [],
        "shown_title_ids": [],
    }


def save_viewer_state(viewer_id: str, state: Dict[str, Any]) -> None:
    """Atomically persist viewer state to disk."""
    path = get_state_path(viewer_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
