# Sonata Model Card

## Limitations and Biases
The system only hears 30-second previews, so every audio feature describes a clip, not the whole song. Tempo, energy, and brightness, come from simple measures (librosa tempo, loudness, spectral centroid) that I never validated against listeners; a slow trap beat can be labeled "slow, low energy" without being calm. 

The star sky is an artistic rendering: pitch is the strongest note detected in a dense mix. The library comes from my own Spotify history and artists related to it, so it reflects one person's taste and mostly popular artists, and "chill" really means what I'd call chill. The library is small, so for some queries there simply aren't enough matches.

The audio effects (speed, reverb, 8D) only apply to the 30-second previews, and 8D needs headphones to be heard as intended.

## Potential Misuse and Prevention
The main risks are leaking credentials and exposing personal adta. I kept API keys in a git-ignored .env, ignored the Spotify token cache, and didn't commit the audio previews, which aren't mine to redistribute. The public repo does reveal which songs I like, so I curated that list before publishing. Free-text queries go to a language model, so I limit them to 500 characters, validate the model's JSON, and drop any song id it returns that wasn't retrieved, logging it as a hallucination. The system shouldn't be used to infer sensitive things about a person from their listening history, and it only works from data its owner chooses to provide.

## Testing Surprises
For "chill songs to study to," adding audio features made the results worse by my own judgement. Ice Spice's "Did it First" and Eladio Carrion's "Sin Frenos" appeared, while Sam Fender's "Rein Me In" and Russ' "Scared" dropped out.

The comparison table still said it improved, because audio match measures similarity to the songs I like, not to the query. I found this by reading the results instead of relying on the numbers.

## AI Collaboration (one suggestion accepted, one rejected or changed)

I used Claude to plan the project, understand each step and write my prompts, and an AI coding assistant to build Sonata from those prompts.

**What I accepted**
- Using RAG as the required AI feature, since my recommender already did retrieval.
- Building in two layers: a terminal pipeline first, then the web sky, so a working, gradable project existed early.
- An agentic workflow for mood labeling. While labeling songs by hand I realized I couldn't label all 379, so I proposed using an AI feature for the rest. I hand-labeled 57 songs, and with Claude's help we designed the flow: a model labels songs using some of my labels as examples, automatic checks flag unlikely labels, the model revises the flagged ones, and anything still uncertain comes to me. I kept some of my labels hidden as a test set to measure agreement with my own judgment.

**What I rejected or changed**
- The coding assistant recommended generating about 50 simulated users for collaborative filtering. I declined: my base project had no collaborative filtering, and invented data would have weakened the project.
- The first ranking blended 70% text similarity with 30% similarity to my taste. Testing showed it pulled "chill songs to study to" toward songs I already like instead of songs that were chill, so I changed it to 85/15 and added a relevance threshold.
- I didn't trust the model's mood labels as they came. After testing, I reviewed them and corrected 20 labels, for example "Did It First": the model called it [chill] and I'd call it [hype].