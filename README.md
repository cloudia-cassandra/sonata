# Sonata

Longer explanations, full tables and the web UI tour are in [docs/details.md](docs/details.md).

## Base Project

Sonata grows out of **VibeCheck 1.0 (Music Recommender Simulation)**, my project from Modules 1-3. VibeCheck was a content-based recommender that scored 20 hand-made placeholder songs against a user profile on genre, mood, energy, valence and acousticness, and returned the top 5 with a reason for every point. It still runs with `python -m src.main`.

## Title and Summary

**Sonata: songs picked from your real taste, explained.** It replaces the placeholder songs with my real Spotify taste, finds each song and related-artist candidates on Deezer, and analyzes their 30-second previews with librosa. It recommends in two ways, content-based (`recommend`) and retrieval-augmented with Gemini (`ask "<request>"`), and runs from the committed data with no login, MP3s or API key.

## Architecture Overview

- **Data pipeline:** Spotify taste → Deezer previews → librosa features (28 per song), star maps and tempo/energy/brightness labels.
- **`recommend`:** cosine similarity on standardized features, each candidate against its closest liked song.
- **`ask`:** MiniLM retrieval (strong matches first, 85% text + 15% taste) keeps 8 songs; Gemini (or mock) picks from them as JSON; a validator drops anything not retrieved, with a cache, an 18-calls-a-day cap and a retrieval-only fallback.
- **Moods:** hand labels, agent labels (label → check → fix → human review) and nearest-neighbor estimates are added to the song descriptions.
- **Web UI and evaluation:** a FastAPI server shows the galaxy and song skies using the same functions; `evaluate` writes `reports/`.

Diagram: [diagrams/architecture.mmd](diagrams/architecture.mmd).

## Setup Instructions

1. Install (Python 3.13):
   ```bash
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
2. Optional, to use Gemini instead of mock mode: copy the example and set `LLM_PROVIDER=gemini`, `LLM_MODEL` and `GEMINI_API_KEY`.
   ```bash
   cp .env.example .env
   ```
3. Ask for songs (the first run downloads the small MiniLM model once):
   ```bash
   python -m sonata ask "upbeat songs for a night drive"
   ```
4. Get content-based recommendations:
   ```bash
   python -m sonata recommend
   ```
5. Open the web UI at http://127.0.0.1:8000:
   ```bash
   python -m sonata.server
   ```
6. Run the tests (mock mode, no API calls):
   ```bash
   pytest
   ```
7. Run the evaluation:
   ```bash
   python -m sonata evaluate
   ```
8. Optional, rebuild the data from your own Spotify taste (needs Spotify credentials in `.env`):
   ```bash
   python -m sonata.fetch_spotify_taste && python -m sonata.build_library && python -m sonata.analyze && python -m sonata.compute_galaxy
   ```
9. Optional, label moods by hand:
   ```bash
   python -m sonata.label
   ```
10. Optional, label the rest with the agent (uses Gemini, more than one day's cap), then review what it flagged:
    ```bash
    python -m sonata.label_agent && python -m sonata.review
    ```

## Sample Interactions

Recorded in mock mode (canned reasons, real retrieval) with the earlier 70/30 ranking, when the score was still called "confidence"; full output in [assets/sample_runs.txt](assets/sample_runs.txt). ♥ = already liked.

```
$ python -m sonata ask 'high energy reggaeton for a party'
1. 2 K 1 6 by Omar Courtz  (confidence 0.68)
   [mock] 2 K 1 6 is Reggaeton with low energy, matching 'high energy reggaeton for a party'.
2. Q U E V A S H A C E R H O Y ? by Omar Courtz ♥  (confidence 0.66)
   [mock] Q U E V A S H A C E R H O Y ? is Reggaeton with medium energy, matching 'high energy reggaeton for a party'.
3. Weeknd<3 by legallyrxx  (confidence 0.65)
   [mock] Weeknd<3 is Reggaeton with low energy, matching 'high energy reggaeton for a party'.
```

```
$ python -m sonata ask 'sad slow songs for a rainy night'
1. Hours In Silence by Drake ♥  (confidence 0.62)
   [mock] Hours In Silence is Rap/Hip Hop with high energy, matching 'sad slow songs for a rainy night'.
2. underwater by FRANKIEONTHEGUITAR ♥  (confidence 0.59)
   [mock] underwater is Pop with medium energy, matching 'sad slow songs for a rainy night'.
3. Drama (feat. Drake) by Roy Woods  (confidence 0.58)
   [mock] Drama (feat. Drake) is R&B with low energy, matching 'sad slow songs for a rainy night'.
```

```
$ python -m sonata ask 'dark rap for the gym'
1. Hours In Silence by Drake ♥  (confidence 0.68)
   [mock] Hours In Silence is Rap/Hip Hop with high energy, matching 'dark rap for the gym'.
2. redrum by 21 Savage  (confidence 0.68)
   [mock] redrum is Rap/Hip Hop with high energy, matching 'dark rap for the gym'.
3. 4 Raws by EsDeeKid ♥  (confidence 0.68)
   [mock] 4 Raws is Rap/Hip Hop with high energy, matching 'dark rap for the gym'.
```

## Design Decisions

- **Spotify for taste, Deezer for audio:** Spotify reflects what I really play but no longer gives new apps previews or audio features, so previews come from Deezer, at the cost of fuzzy matching that must agree on the artist to avoid cover versions.
- **My own librosa features:** every preview gets the same 28 numbers (tempo, loudness, brightness, timbre, harmony), but energy and brightness labels are thirds of my library, so "high energy" means loud *for my songs*, not in absolute terms.
- **The LLM only picks from 8 retrieved songs:** every pick is checkable and a hallucinated song is easy to drop, but the model can never suggest a good song that retrieval missed.
- **Built for about 20 free requests a day:** a disk cache, an 18-call daily cap, mock mode by default and a retrieval-only fallback keep it free and crash-proof, but real Gemini behavior gets tested far less often.
- **Relevance first, taste second, scored by retrieval similarity:** taste (15%) only reorders songs that already match the request, and the score shown is retrieval similarity rather than a "confidence" the model can't calibrate, which makes answers more on-request but less personal.

## Testing Summary

| Check | Result |
|---|---|
| Unit tests (`pytest`, mock mode) | 150 passing |
| RAG evaluation: grounding, 10 queries × 3 runs | 10/10 |
| Consistency across runs | 10/10 (mock, so 100% by construction) |
| Graceful handling of empty, too-long and nonsense input | 10/10 |
| Hallucinated songs dropped by the validator | 0 |

- **What worked:** the guardrails: every pick was grounded in retrieval, bad input was rejected with a message, and nonsense got the fallback instead of a model answer.
- **What didn't:** the evaluation ran in mock mode, so real Gemini consistency is still unmeasured (it needs more calls than one day's cap); and the 70/30 audio blend let taste override the request.
- **What I learned:** text descriptions can't express mood, so vague requests like "chill songs to study to" matched weakly, which is why mood labels came next.

## RAG Enhancement

Audio re-ranking blends in how much a song sounds like my liked songs (measured with the original 70/30 blend; now 85/15, among strong matches only):

```
"dark rap for the gym"
before (text only):     Runnin — 21 Savage · Scared — Russ ♥ · Redbone — Childish Gambino
after (audio re-rank):  Hours In Silence — Drake ♥ · redrum — 21 Savage · 4 Raws — EsDeeKid ♥
audio match 0.61 → 0.88, liked songs 1 → 2
all 8 answered queries: audio match 0.65 → 0.79, 15% overlap with the text-only picks
```
