# Sonata Model Card

## Limitations and Biases
The system only hears 30-second previews, so every audio feature describes a clip, not the whole song. Tempo, energy, and brightness, come from simple measures (librosa tempo, loudness, spectral centroid) that I never validated against listeners; a slow trap beat can be labeled "slow, low energy" without being calm. 

The star sky is an artistic rendering: pitch is the strongest note detected in a dense mix. The library comes from my own Spotify history and artists related to it, so it reflects one person's taste and mostly popular artists, and "chill" really means what I'd call chill. The library is small, so for some queries there simply aren't enough matches.

## Potential Misuse and Prevention
The main risks are leaking credentials and exposing personal adta. I kept API keys in a git-ignored .env, ignored the Spotify token cache, and didn't commit the audio previews, which aren't mine to redistribute. The public repo does reveal which songs I like, so I curated that list before publishing. Free-text queries go to a language model, so I limit them to 500 characters, validate the model's JSON, and drop any song id it returns that wasn't retrieved, logging it as a hallucination. The system shouldn't be used to infer sensitive things about a person from their listening history, and it only works from data its owner chooses to provide.

## Testing Surprises
For "chill songs to study to," adding audio features made the results worse by my own judgement. Ice Spice's "Did it First" and Eladio Carrion's "Sin Frenos" appeared, while Sam Fender's "Rein Me In" and Russ' "Scared" dropped out.

The comparison table still said it improved, because audio match measures similarity to the songs I like, not to the query. I found this by reading the results instead of relying on the numbers.

## AI Collaboration (one suggestion accepted, one rejected or changed)
I used Claude to plan the project and understand each step, to decide how to prompt along the way, and I used an AI coding assistant to take along the process of building Sonata.

A suggestion I accepted: using RAG as the required AI feature, since my recommender already did retrieval. I also accepted the plan to build in 2 layers (a terminal pipeline first, the web sky second), which meant a working, gradable project existed early.

Suggestions I rejected or changed: the coding assistant recommended creating about 50 simulated users for collaborative filtering. I declined, because my base project had no collaborative filtering, and invented data would have weakened the project. I also changed the first jitter settings for the star sky; they were too small to matter, so the sky still showed straight rows, and I asked for a more natural scatter. 
