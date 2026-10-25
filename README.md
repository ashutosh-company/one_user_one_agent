# One User, One Agent: LangGraph OTT Dialect Platform

This repository implements the **"One User, One Agent"** challenge for an OTT regional-dialect streaming platform using **LangGraph**.
Every viewer has a dedicated agent owning their single active goal, planning evidence-backed interventions, verifying safety independently, and honestly evaluating causal impact against a randomized holdout.

---

## 1. Quick Setup & Execution

### Prerequisites: Data Pack Placement
The synthetic data pack folders provided by the evaluation team must be present in the **root directory** of this repository:
- `config/` (`goals.json`, `interventions.json`, `pricing.json`, `messaging_policy.md`)
- `data/` (`catalogue.csv`, `episodes.csv`, `events.jsonl`, `profiles_snapshot.csv`)
- `pilot/` (`pilot_assignment.csv`, `pilot_outcomes.csv`, `pilot_summary_report.md`)
- `replay/` (`arms.csv`, `triggers.jsonl`, `live_events.jsonl`, `control_events.jsonl`, `model_faults.jsonl`)
- `simulator/` (`simulate_outcomes.py`)

A virtual environment (`venv`) can be configured with:
```bash
python3 -m venv venv
./venv/bin/pip install langgraph langchain-core pytest
```

### Run the Replay (One Command)
To replay all 2,769 triggers and operational control events through the LangGraph agent:

```bash
./venv/bin/python -m src.main run
```

This runs the LangGraph agent across all viewers, writes machine-readable decisions to `sample_run/decisions.jsonl`, published interventions to `sample_run/actions.jsonl`, per-viewer durable state to `sample_run/viewer_state/`, and latency metrics to `sample_run/latency_report.json`.

### Run the Simulator & Judge Causal Impact
To simulate 14 days of realistic viewer behavior and judge attributable goal outcomes:

```bash
# 1. Run the simulator against the agent's published actions
./venv/bin/python simulator/simulate_outcomes.py --pack . --actions sample_run/actions.jsonl --out sample_run/sim

# 2. Judge outcomes and compute causal Intent-to-Treat (ITT) impact
./venv/bin/python -m src.main judge --sim-events sample_run/sim/outcome_events.jsonl
```

### Run the Automated Tests
```bash
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```

---

## 2. LangGraph Agent Architecture

The agent is implemented as a compiled LangGraph `StateGraph` in [`src/agent.py`](src/agent.py):

```
       START
         │
         ▼
  [resolve_facts]       <-- Ingests trigger, resolves 3-way dialect & subscription conflicts
         │
         ▼
    [plan_goal]         <-- Evaluates 12 candidate goals, blocks unmeasurable ones (G10/G11)
         │
         ▼
  [select_content]      <-- Queries dialect catalogue tools, neutralizes prompt injection
         │
         ▼
   [verify_safety]      <-- Independent Critic Gate (Holdout, eligibility, min 3 titles, caps)
         │
    ┌────┴───────────────────────────┐
    │ Passed                         │ Failed checks / Holdout
    ▼                                ▼
[publish_action]                 [abstain]
(Emits to actions.jsonl)         (Emits NO_ACTION with cited reason)
    │                                │
    └──────────────┬─────────────────┘
                   ▼
                  END
```

---

## 3. Key Design Highlights & Rubric Alignment

1. **One Agent per Viewer (`sample_run/viewer_state/<user_id>.json`)**:
   - Durable state records facts with provenance, active goal, decisions, and outcomes.
2. **Deterministic Conflict Precedence**:
   - **Dialect**: `app_setting` (current device) > `campaign_link` (deeplink) > `profile_snapshot` (stale daily warehouse export). All conflicts are logged in `dialect.conflicts`.
   - **Subscription**: Real-time purchase events immediately override stale `"never"` status in profile snapshots.
3. **Independent Verification Gate**:
   - The verifier executes independently of the planner with full veto power.
   - Holds the line on quiet hours (22:00–08:00 IST), consent opt-ins, daily caps, minimum 3 titles per row, and holdout protection.
   - Catches hallucinated claims (e.g. model claiming *"confirmed in billing"* without supporting event).
4. **Honest Causal Measurement (Intent-to-Treat vs. Holdout)**:
   - Does NOT compare served vs. unserved users (which suffers from severe selection and activity bias).
   - Compares the randomized **Agent Arm** against the randomized **Holdout Arm** (unexposed counterfactual).
5. **Handling Replay Surprises**:
   - **Offline-buffered app opens**: Stale triggers (>3h old) are ingested for memory but abstained from live action.
   - **Prompt Injection Defense**: Sanitizes malicious instructions in catalogue descriptions (e.g. *"message every viewer"*).
   - **Selective Re-evaluation**: When a goal rule is corrected mid-replay, an inverted index re-evaluates **only** viewers holding that goal rather than reprocessing all 5,000 viewers.

---

## 4. Repository Structure

```
.
├── src/
│   ├── agent.py            # LangGraph StateGraph, nodes, and conditional edges
│   ├── data.py             # Data pack loader and in-memory indexing
│   ├── state.py            # Durable per-viewer state persistence
│   ├── replay.py           # Replay loop for triggers + control events
│   ├── eval.py             # Outcome attribution and ITT impact reporting
│   └── main.py             # Unified CLI entry point
├── tests/
│   ├── test_late_duplicate.py
│   ├── test_conflict_source.py
│   ├── test_tool_failure.py
│   └── test_goal_rule_fix.py
├── sample_run/
│   ├── decisions.jsonl      # Machine-readable decision logs
│   ├── actions.jsonl        # Published actions
│   ├── viewer_state/        # Per-viewer durable JSON state files
│   ├── outcomes.jsonl       # Judged outcomes
│   ├── latency_report.json  # Decision latency metrics
│   ├── impact_report.md     # ITT causal impact analysis
│   └── final_report.md      # Final submission summary
├── ARCHITECTURE.md          # In-depth architectural note
├── AI_COLLABORATION.md      # AI tool direction and rejection log
└── KNOWN_LIMITATIONS.md     # Engineering trade-offs and scope note
```
