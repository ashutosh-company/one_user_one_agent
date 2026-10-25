"""
Data loader and in-memory indexer for the OTT Dialect Platform data pack.
Provides clean, fast access to catalogue, profiles, historical events, and replay streams.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(rel_path: str) -> Any:
    path = PROJECT_ROOT / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(rel_path: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / rel_path
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def load_csv(rel_path: str) -> List[Dict[str, str]]:
    path = PROJECT_ROOT / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class DataPack:
    """Holds parsed and indexed data for fast, O(1) agent lookups."""
    def __init__(self):
        print("Loading and indexing data pack...")
        self.goals = load_json("config/goals.json").get("goals", [])
        self.interventions = load_json("config/interventions.json")
        self.pricing = load_json("config/pricing.json")
        
        # Indexed by user_id
        self.arms = {r["user_id"]: r["arm"] for r in load_csv("replay/arms.csv")}
        
        # Profiles (keep most recent if duplicates exist)
        self.profiles: Dict[str, Dict[str, str]] = {}
        for row in load_csv("data/profiles_snapshot.csv"):
            uid = row.get("user_id")
            if uid:
                existing = self.profiles.get(uid)
                if not existing or row.get("snapshot_at", "") > existing.get("snapshot_at", ""):
                    self.profiles[uid] = row

        # Historical events by viewer
        self.events_by_viewer: Dict[str, List[Dict[str, Any]]] = {}
        for event in load_jsonl("data/events.jsonl"):
            uid = event.get("user_id")
            if uid:
                self.events_by_viewer.setdefault(uid, []).append(event)
        for uid in self.events_by_viewer:
            self.events_by_viewer[uid].sort(key=lambda e: e.get("received_at", ""))

        # Live events arriving during replay
        self.live_events_by_viewer: Dict[str, List[Dict[str, Any]]] = {}
        for event in load_jsonl("replay/live_events.jsonl"):
            uid = event.get("user_id")
            if uid:
                self.live_events_by_viewer.setdefault(uid, []).append(event)

        # Catalogue index (by title_id and by dialect)
        self.catalogue: Dict[str, Dict[str, Any]] = {}
        self.catalogue_by_dialect: Dict[str, List[Dict[str, Any]]] = {}
        for item in load_csv("data/catalogue.csv"):
            tid = item["title_id"]
            d = item.get("dialect", "d1")
            self.catalogue[tid] = item
            self.catalogue_by_dialect.setdefault(d, []).append(item)

        # Replay triggers & control events
        self.triggers = load_jsonl("replay/triggers.jsonl")
        self.control_events = load_jsonl("replay/control_events.jsonl")
        self.model_faults = {f["trigger_event_id"]: f for f in load_jsonl("replay/model_faults.jsonl")}

        # Dynamic state updated by control events
        self.withdrawn_titles = set()
        self.corrected_goals = {}
        print(f"Loaded: {len(self.profiles)} profiles, {len(self.events_by_viewer)} active viewers, {len(self.catalogue)} titles.")


# Global singleton instance for clean reuse across graph nodes
_PACK_INSTANCE = None

def get_data_pack() -> DataPack:
    global _PACK_INSTANCE
    if _PACK_INSTANCE is None:
        _PACK_INSTANCE = DataPack()
    return _PACK_INSTANCE
