# Sonata

Longer explanations, full tables and the web UI tour are in [docs/details.md](docs/details.md).

## Base Project

Sonata grows out of **VibeCheck 1.0 (Music Recommender Simulation)**, my project from Modules 1-3. VibeCheck was a content-based recommender that scored 20 hand-made placeholder songs against a user profile on genre, mood, energy, valence and acousticness, and returned the top 5 with a reason for every point. It still runs with `python -m src.main`.

## Title and Summary

**Sonata: songs picked from your real taste, explained.** It replaces the placeholder songs with my real Spotify taste, finds each song and related-artist candidates on Deezer, and analyzes their 30-second previews with librosa. It recommends in two ways, content-based (`recommend`) and retrieval-augmented with Gemini (`ask "<request>"`), and runs from the committed data with no login, MP3s or API key.

## Architecture Overview

A request first passes an input guardrail that rejects empty or over-500-character text.
The retriever then picks the 8 songs whose descriptions best match it (MiniLM text similarity, with 15% weight on sounding like my liked songs); if nothing scores above 0.40, Sonata skips the model and returns the closest songs with a notice.
The LLM (Gemini, or mock mode by default; cached and capped at 18 calls a day) picks songs from those 8 and explains each one.
A validator drops any song that wasn't retrieved or is duplicated, and an API error, exhausted quota or too few valid picks falls back to retrieval-only results.
The output is the songs with their reasons and retrieval similarity, and every request is logged; pytest and the evaluation script check grounding, consistency and bad input, and I review the reports and logs to tune the retriever.
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
- **Audio effects in the browser, not in files:** speed, reverb and 8D use the Web Audio API on the preview as it plays, so nothing needs processing or storing, but nothing can be saved or exported either.

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

## Agentic Workflow: Mood Labeling

`python -m sonata.label_agent` labels every song I haven't labeled by hand, in four steps:

1. **Label:** Gemini labels batches of 25 songs (title, artist, genre, tempo/energy/brightness labels, stars per second) with 1-3 moods, a confidence and a reason, using 15 of my hand labels as examples. The JSON is validated, moods outside the vocabulary are dropped, and a failed batch is retried once.
2. **Check:** without the model, a label is flagged if its moods contradict (chill with hype), its confidence is below 0.6, the song's numbers fall far outside my own labels' range for that mood, or none of its moods appear among its 5 most similar-sounding labeled songs.
3. **Fix:** flagged songs go back to Gemini with the flags and evidence, to revise or confirm; the check runs again.
4. **Human review:** songs still flagged wait for me in `python -m sonata.review`, where I accept or replace the moods. Hand labels always win over agent labels.

Every step for every song is traced in `logs/agent_trace.jsonl`. The other hand-labeled songs are held out to score the model with and without examples, after the checks, and against a nearest-neighbor estimate, in `reports/label_eval.md`. A full run needs more than one day's 18 calls, so it stops when the budget runs out and resumes from the cache the next day.

## Audio Effects Mixer

- **What it does:** changes how a song's preview sounds in its sky: speed (0.5-1.5×), reverb (0-100%) and 8D (sound circling your head), with presets Normal, Slowed + Reverb, Sped up and 8D.
- **How it works:** the Web Audio API in `web/fx.js`: the playback rate with pitch correction off (slower is lower, like "slowed" edits), a convolver with a generated 2.5-second echo for reverb, and a stereo panner swept left-right every 8 seconds for 8D. The stars stay in sync at any speed.
- **How to use it:** start `python -m sonata.server`, open a song's sky, press Play, and use the panel at the bottom.
- **Limitation:** it only changes playback of the 30-second previews in the browser and can't export audio.

## Reflection

Building Sonata taught me that a number going up doesn't mean the system got better. My RAG comparison showed audio match and confidence both improving, but when I read the actual picks for "chill songs to study to," they were worse than before. Those metrics measured similarity to my taste and to the query text, not whether a song was chill. That pushed me to add my own mood labels and measure against them. I also learned to design around constraints: with a 20-requests-a-day model limit, caching, a mock mode and a daily budget guard became part of the system. And using AI well meant treating its output as a draft to test against my own judgment, in small steps, instead of trusting it because it ran.
