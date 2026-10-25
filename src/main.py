"""
CLI Entry Point for the LangGraph-based "One User, One Agent" system.

Usage:
  python -m src.main run
  python -m src.main judge [--sim-events path/to/outcome_events.jsonl]
"""
import sys
import argparse
from pathlib import Path

from src.replay import run_replay
from src.eval import judge_outcomes, compute_impact_report


def main():
    parser = argparse.ArgumentParser(description="One User, One Agent – LangGraph Implementation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    subparsers.add_parser("run", help="Replay all triggers through the LangGraph agent")

    # Subcommand: judge
    judge_parser = subparsers.add_parser("judge", help="Judge outcomes and compute causal impact")
    judge_parser.add_argument("--sim-events", default=None, help="Path to outcome_events.jsonl from simulator")

    args = parser.parse_args()

    if args.command == "run":
        run_replay()
    elif args.command == "judge":
        sim_path = Path(args.sim_events) if args.sim_events else None
        outcomes = judge_outcomes(sim_path)
        compute_impact_report(outcomes)


if __name__ == "__main__":
    main()
