# Sonata RAG evaluation

Generated 2026-11-15 23:09 with provider `mock`; 10 queries × 3 runs.

| # | Query | Kind | Grounding | Consistency | Graceful | Avg confidence | Sources |
|---|---|---|---|---|---|---|---|
| 1 | high energy reggaeton for a party | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.67 | mock |
| 2 | chill songs to study to | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.53 | mock |
| 3 | sad slow songs for a rainy night | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.60 | mock |
| 4 | dark rap for the gym | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.68 | mock |
| 5 | songs like Bad Bunny | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.63 | mock |
| 6 | bright happy pop | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.52 | mock |
| 7 | music for a late night drive | normal | ✅ pass | ✅ pass (100%) | ✅ pass | 0.60 | mock |
| 8 | *(empty)* | bad_input | ✅ pass | ✅ pass (100%) | ✅ pass | — | rejected |
| 9 | asdkjh qwpoeiru zzxv | nonsense | ✅ pass | ✅ pass (100%) | ✅ pass | 0.38 | fallback |
| 10 | play something play something play so… (599 chars) | bad_input | ✅ pass | ✅ pass (100%) | ✅ pass | — | rejected |
| | **Total** | | **10/10** | **10/10** | **10/10** | **0.57** | |

Hallucinated songs dropped by the validator: 0. Exceptions: 0.

**Checks.** Grounding: every recommended song was in that run's retrieved set. Consistency: mean pairwise overlap of the recommended songs across the 3 runs is at least 60%. Graceful: normal requests get 3-5 songs (a fallback must say why), empty or over-500-character input is rejected with a message, and nonsense gets a low-confidence notice instead of a model answer. Confidence is the retrieval score (0-1) of each recommended song.

> **Note:** every run came from the mock provider, the response cache, a fallback or a rejection, which are all deterministic, so the consistency column is 100% by construction here. Run `python -m sonata evaluate --fresh` with Gemini to measure the model's real run-to-run consistency.

## Messages shown to the user

- **#8:** Please type what you'd like to hear, for example "upbeat songs for a night drive".
- **#9:** Nothing in your library closely matches "asdkjh qwpoeiru zzxv" (best match 0.26); here are the closest songs, with low confidence.
- **#10:** Your request is 599 characters long; please keep it to 500 characters or fewer.
