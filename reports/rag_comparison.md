# Text-only vs. text + audio retrieval

Generated 2026-11-15 23:09 with provider `mock`. **Before:** `use_audio=False`, songs ranked by text similarity only. **After:** `use_audio=True`, the top 24 text matches re-ranked by 70% text + 30% audio similarity to your closest liked song.

*Audio match* is the recommended songs' average audio similarity (cosine on standardized features) to their closest liked song, computed the same way for both columns; ♥ marks songs you already like.

| # | Query | Before: picks | After: picks | Overlap | Confidence before → after | Audio match before → after | ♥ before → after |
|---|---|---|---|---|---|---|---|
| 1 | high energy reggaeton for a p… (33 chars) | Weeknd<3 — legallyrxx<br>BAKANO — legallyrxx<br>RIGHT NOW — Lucho RK | 2 K 1 6 — Omar Courtz<br>Q U E V A S H A C E R H O Y ? — Omar Courtz ♥<br>Weeknd<3 — legallyrxx | 20% | 0.65 → 0.67 | 0.63 → 0.72 | 0 → 1 |
| 2 | chill songs to study to | Did It First — Ice Spice ♥<br>Rein Me In — Sam Fender ♥<br>Scared — Russ ♥ | Did It First — Ice Spice ♥<br>Help Me Lose My Mind — Disclosure<br>Sin Frenos — Eladio Carrion ♥ | 20% | 0.45 → 0.53 | 0.64 → 0.72 | 3 → 2 |
| 3 | sad slow songs for a rainy ni… (32 chars) | Nights — Frank Ocean ♥<br>the cure — Olivia Rodrigo<br>NO CRY — Kapo ♥ | Hours In Silence — Drake ♥<br>underwater — FRANKIEONTHEGUITAR ♥<br>Drama (feat. Drake) — Roy Woods | 0% | 0.52 → 0.60 | 0.59 → 0.84 | 2 → 2 |
| 4 | dark rap for the gym | Runnin — 21 Savage<br>Scared — Russ ♥<br>Redbone — Childish Gambino | Hours In Silence — Drake ♥<br>redrum — 21 Savage<br>4 Raws — EsDeeKid ♥ | 0% | 0.62 → 0.68 | 0.61 → 0.88 | 1 → 2 |
| 5 | songs like Bad Bunny | Party — Bad Bunny ♥<br>Moscow Mule — Bad Bunny ♥<br>Agosto — Bad Bunny ♥ | Andrea — Bad Bunny ♥<br>Efecto — Bad Bunny ♥<br>Party — Bad Bunny ♥ | 20% | 0.57 → 0.63 | 0.68 → 0.79 | 3 → 3 |
| 6 | bright happy pop | LEGACY — PIXY ♥<br>Ella — Mishnrz ♥<br>Shiny — Easykid ♥ | LEGACY — PIXY ♥<br>la culpa (track 11) — alleh ♥<br>Antagonista — NSQK ♥ | 20% | 0.45 → 0.52 | 0.64 → 0.77 | 3 → 3 |
| 7 | music for a late night drive | Nights — Frank Ocean ♥<br>Blind — OVI WOOD<br>Late Night Talking — Harry Styles ♥ | Static — Steve Lacy ♥<br>This Is America — Childish Gambino<br>JACKIE BROWN — Brent Faiyaz ♥ | 0% | 0.53 → 0.60 | 0.68 → 0.83 | 2 → 2 |
| 8 | *(empty)* | *rejected* | *rejected* | 100% | — → — | — → — | 0 → 0 |
| 9 | asdkjh qwpoeiru zzxv | Tu No Vive Asi (feat. Mambo Kingz & DJ Luian) — Arcángel<br>ZABKA — Azzfit<br>RANA POSTRZALOWA — Azzfit<br>Mopao aye (feat. MC YOSHI, Mauvais Djo & Kokosvoice) — TRIANGLE DES BERMUDES<br>7 Lágrimas (Remix) — Soge Culebra | ZABKA — Azzfit<br>Tu No Vive Asi (feat. Mambo Kingz & DJ Luian) — Arcángel<br>RANA POSTRZALOWA — Azzfit<br>Cmon (LATIN MAFIA & Fred edit) — Fred again.. ♥<br>Q U E V A S H A C E R H O Y ? — Omar Courtz ♥ | 43% | 0.22 → 0.38 | 0.75 → 0.80 | 0 → 2 |
| 10 | play something play something… (599 chars) | *rejected* | *rejected* | 100% | — → — | — → — | 0 → 0 |

**Averages over the 8 queries that returned songs:** overlap 15%, confidence 0.50 → 0.57, audio match 0.65 → 0.79, liked songs per answer 1.8 → 2.1.

Confidence is each column's own ranking score (text similarity before, the 70/30 blend after), so the two confidence numbers are not on exactly the same scale.
