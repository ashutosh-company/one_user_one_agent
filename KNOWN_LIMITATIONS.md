# Known Limitations & Engineering Trade-Offs

In accordance with the 10–12 hour challenge scope, effort was prioritized toward establishing an ironclad, auditable **decide, act, and measure loop** rather than building complex peripheral services.

The following areas were intentionally left bounded or simplified:

---

## 1. Scope & Action Channels
- **Focus on Personalized Home Page**:
  - The primary published action is `HOME_PAGE` layout personalization, as this reaches active app-openers immediately with zero external delivery risk.
  - While verification rules for `PUSH` and `WHATSAPP` (quiet hours, daily frequency caps, consent opt-in checks) are fully implemented and tested in the verifier, automated message dispatch was kept minimal during replay to avoid spamming synthetic channels without confirmed device tokens.

## 2. In-Memory Graph Indexing vs. Distributed KV Store
- **Current Architecture**: The agent indexes the catalogue, profiles, and event streams in memory at startup. This enables sub-10ms decision latencies and O(1) lookups during the replay.
- **Production Extension**: At 10 million viewers and 40 dialects, in-memory Python dictionaries would be replaced by Redis / Aerospike for state caching and a persistent vector / relational datastore with distributed sharding.

## 3. Heuristic & Rule-Based Ranking vs. Fine-Tuned Semantic Model
- **Current Implementation**: Title selection uses goal-directed filtering (e.g. prioritizing unfinished series for `G07`, new releases for `G08`, and free-trial friendly titles for onboarding) paired with deterministic safety sanitization.
- **Production Extension**: A two-stage collaborative filtering or embedding-based ranker (e.g. Two-Tower model) fine-tuned on user watch history could be plugged into the `select_content_node` without altering the surrounding verification or state architecture.

## 4. Episode Completion Tracking
- **Limitation**: The event schema only emits `episode_completed` (90%+ watched) without fine-grained per-minute scrubbing data.
- **Assumption Made**: For goal `G07` (`resume_series_3d`), the agent assumes unwatched episodes remain if `play_started` was logged for a series title and no `episode_completed` exists.

## 5. Mock Mode Default for Evaluators
- **Design Decision**: Live network LLM calls are disabled by default to ensure evaluators can run the entire 2,769-trigger replay and pytest suite offline without requiring personal API keys or incurring external API bills.
