// The song sky. Every star is a note onset from sonata.analyze:
// x = time, y = pitch (MIDI), radius = loudness. Stars light up as the playhead passes.
import { appear, circle, clamp, font, ink, reduceMotion, seededRandom, theme } from "./util.js";

const PREVIEW_SECONDS = 30;
const FLASH_MS = 700;          // how long a newly lit star glows
const SEEK_JUMP = 0.5;         // seconds per frame; bigger moves are seeks, which light stars without flashing
const AXIS_HEIGHT = 34;        // px below the plot for the second markers
const MIN_PITCH_SPAN = 24;     // semitones, so a narrow melody isn't stretched to fill the sky
const SPECKS = 170;
const INTRO_MS = 420;         // a newly loaded sky fades in...
const INTRO_SWEEP_MS = 650;   // ...with its stars appearing left to right over this long
const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

// A natural-looking sky. Pitch is quantized to semitones and loudness is often similar, so a
// repeated note would draw as a ruler-straight row of identical stars. Each star gets a fixed
// scatter instead, seeded by the song and the star's index, so it's the same on every visit.
// Only the drawing moves: lighting, the playhead and the note name use each star's real t and pitch.
// Setting SKY_Y_SCATTER, SKY_DRIFT, SKY_X_JITTER, SKY_SIZE_JITTER, SKY_BRIGHTNESS_JITTER and
// SKY_TWINKLE to 0 (and SKY_ARC_OPACITY to 1) gives back the strict grid exactly.
const SKY_Y_SCATTER = 0.035;        // vertical scatter, a fraction of the sky's height; heavy-tailed:
                                    // average of two randoms, and ~10% of stars get twice as much
const SKY_CROWD_BOOST = 2.0;        // up to this multiple of extra scatter for crowded pitches (many stars
                                    // within ±1 semitone): 1x at <=3 neighbors, full boost at >=10
const SKY_DRIFT = 0.04;             // slow vertical drift along time, a fraction of the sky's height
                                    // (three sine waves, ~7.3 s, 3.1 s and 1.7 s, seeded phases)
const SKY_X_JITTER = 0.2;           // horizontal scatter, at most ± this many seconds' worth of pixels
const SKY_SIZE_JITTER = 0.55;       // size multiplier 1 - j + 2j*random^1.8: skewed toward small stars
const SKY_BRIGHTNESS_JITTER = 0.4;  // lit stars shine at between 1 - this and 1 (0.6-1.0)
const SKY_TWINKLE = 0.15;           // slow opacity shimmer on unlit stars (each 3-8 s, own phase); lit stars hold steady
const SKY_SHOW_ARC = true;          // the dashed pitch-trend curve across the sky
const SKY_ARC_OPACITY = 0.5;        // multiplier on that curve's opacity
const SKY_MIN_GAP = 1.5;            // px kept between neighboring stars after scattering...
const SKY_MAX_NUDGE = 6;            // ...by pushing the later one aside at most this far

const CROWD_FROM = 3;               // neighbors within ±1 semitone at which the crowd boost starts...
const CROWD_FULL = 10;              // ...and reaches its full strength
const TAIL_SHARE = 0.1;             // share of stars with the doubled, heavy-tail scatter
const DRIFT_PERIODS = [7.3, 3.1, 1.7];
const POSITION_SCATTER = SKY_Y_SCATTER > 0 || SKY_DRIFT > 0 || SKY_X_JITTER > 0;

let ctx, audio, ui;

const state = {
  track: null,
  stars: [],
  litAt: [],              // per star: null = dark, a timestamp = when it lit, -Infinity = lit without a flash
  specks: [],
  trend: null,            // quadratic pitch trend, drawn as the dashed arc
  duration: PREVIEW_SECONDS,
  pitchLo: 36,
  pitchHi: 84,
  manual: false,          // true: no playable preview, the scrub bar drives the playhead
  lastT: 0,
  litCount: -1,
  timeText: "",
  clock: { audioT: 0, wall: 0 },
  plot: null,
  rMax: 7,
  loadToken: 0,
  dragging: false,
  shownAt: Infinity,      // when the current star map arrived (drives the intro)
  scatter: [],            // per star, fixed at load: {dY, dT, sizeMul, brightMul, twinkle*, nudgeAngle}
  placed: [],             // per star, drawn position {x, y, r}; recomputed only when the layout changes
  placedKey: "",
};

// ---------- helpers ----------

// Restart an element's CSS entrance animation (the header when the song changes)
function replayEntrance(element) {
  element.classList.remove("enter");
  void element.offsetWidth;
  element.classList.add("enter");
}

function noteName(midi) {
  return NOTE_NAMES[midi % 12] + (Math.floor(midi / 12) - 1);
}

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// Least-squares fit p = c0 + c1*u + c2*u^2 over u in [0, 1], weighted by loudness
function quadraticTrend(stars, duration) {
  if (stars.length < 3) return null;
  const s = [0, 0, 0, 0, 0];
  const r = [0, 0, 0];
  for (const star of stars) {
    const u = star.t / duration;
    const w = 0.2 + star.size;
    let uk = 1;
    for (let k = 0; k < 5; k++) {
      s[k] += w * uk;
      if (k < 3) r[k] += w * uk * star.pitch;
      uk *= u;
    }
  }
  const m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
  const det = (a) =>
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const d = det(m);
  if (Math.abs(d) < 1e-9) return null;
  // Cramer's rule: replace column i with r
  return [0, 1, 2].map((i) => det(m.map((row, j) => row.map((v, k) => (k === i ? r[j] : v)))) / d);
}

const xAt = (t) => state.plot.x + (t / state.duration) * state.plot.w;
const yAt = (pitch) =>
  state.plot.y + state.plot.h - ((pitch - state.pitchLo) / (state.pitchHi - state.pitchLo)) * state.plot.h;
const radiusOf = (size) => 1.3 + Math.sqrt(clamp(size, 0, 1)) * (state.rMax - 1.3);
const timeAtX = (x) => clamp(((x - state.plot.x) / state.plot.w) * state.duration, 0, state.duration);

// ---------- natural scatter ----------

// How crowded each star's pitch is: 1 for notes with few neighbors within ±1 semitone,
// rising to SKY_CROWD_BOOST for pitches the song keeps repeating
function crowdBoosts(stars) {
  const perPitch = new Map();
  for (const star of stars) perPitch.set(star.pitch, (perPitch.get(star.pitch) || 0) + 1);
  return stars.map((star) => {
    const near = (perPitch.get(star.pitch - 1) || 0) + perPitch.get(star.pitch) + (perPitch.get(star.pitch + 1) || 0) - 1;
    return 1 + (SKY_CROWD_BOOST - 1) * clamp((near - CROWD_FROM) / (CROWD_FULL - CROWD_FROM), 0, 1);
  });
}

// Computed once per song. Every star draws the same nine random numbers whatever the settings
// are, so tuning one constant never reshuffles the others. dY is a fraction of the sky's height.
function scatterStars(stars, deezerId) {
  const seed = deezerId % 4294967296;
  const songRandom = seededRandom(seed ^ 0x5bd1e995);
  const phases = DRIFT_PERIODS.map(() => songRandom() * Math.PI * 2);
  // Slow drift along the song: neighbors in time drift together, like a band of the Milky Way
  const drift = (t) =>
    (SKY_DRIFT / DRIFT_PERIODS.length) *
    DRIFT_PERIODS.reduce((sum, period, k) => sum + Math.sin((2 * Math.PI * t) / period + phases[k]), 0);
  const crowd = crowdBoosts(stars);

  return stars.map((star, i) => {
    const random = seededRandom(seed ^ Math.imul(i + 1, 0x9e3779b1));
    const [a, b, c, d, e, f, g, h, k] = Array.from({ length: 9 }, random);
    const tail = c < TAIL_SHARE ? 2 : 1;
    return {
      dY: (a + b - 1) * SKY_Y_SCATTER * tail * crowd[i] + drift(star.t),
      dT: (d * 2 - 1) * SKY_X_JITTER,
      sizeMul: 1 - SKY_SIZE_JITTER + 2 * SKY_SIZE_JITTER * Math.pow(e, 1.8),
      brightMul: 1 - f * SKY_BRIGHTNESS_JITTER,
      twinklePhase: g * Math.PI * 2,
      twinklePeriod: 3000 + h * 5000,              // ms
      nudgeAngle: (k * 2 - 1) * Math.PI,           // which way to push a star that lands exactly on another
    };
  });
}

// Screen position and radius of every star, with the scatter applied and close neighbors
// separated. Cached; redone only when something that moves stars on screen changes.
function placedStars() {
  const { plot, stars, scatter } = state;
  const key = [plot.x, plot.y, plot.w, plot.h, state.duration, state.pitchLo, state.pitchHi, state.rMax, stars.length].join();
  if (key === state.placedKey) return state.placed;

  // Keep every star inside the sky. A star whose true position already touches an edge (e.g. a
  // note at 0 s) may stay where it is, so with no scatter nothing moves at all.
  const bounds = stars.map((star, i) => {
    const r = radiusOf(star.size) * scatter[i].sizeMul;
    const x0 = xAt(star.t);
    const y0 = yAt(star.pitch);
    return {
      r,
      left: Math.min(plot.x + r, x0), right: Math.max(plot.x + plot.w - r, x0),
      top: Math.min(plot.y + r, y0), bottom: Math.max(plot.y + plot.h - r, y0),
    };
  });
  const keepInside = (p, b) => {
    p.x = clamp(p.x, b.left, b.right);
    p.y = clamp(p.y, b.top, b.bottom);
  };
  const placed = stars.map((star, i) => {
    const p = { x: xAt(star.t + scatter[i].dT), y: yAt(star.pitch) - scatter[i].dY * plot.h, r: bounds[i].r };
    keepInside(p, bounds[i]);
    return p;
  });

  // With no positional scatter, keep the original look exactly (repeated notes overlap there too)
  if (POSITION_SCATTER) {
    const pxPerSecond = plot.w / state.duration;
    const reach = 2 * state.rMax * (1 + SKY_SIZE_JITTER) + SKY_MIN_GAP + SKY_MAX_NUDGE + 2 * SKY_X_JITTER * pxPerSecond;
    for (let i = 1; i < placed.length; i++) {
      const p = placed[i];
      // Stars are in time order, so only the few just before this one can be close to it
      for (let j = i - 1; j >= 0 && (stars[i].t - stars[j].t) * pxPerSecond <= reach; j--) {
        const q = placed[j];
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const dist = Math.hypot(dx, dy);
        const need = p.r + q.r + SKY_MIN_GAP;
        if (dist >= need) continue;
        const push = Math.min(need - dist, SKY_MAX_NUDGE);
        const angle = dist > 0.01 ? Math.atan2(dy, dx) : scatter[i].nudgeAngle;
        p.x += Math.cos(angle) * push;
        p.y += Math.sin(angle) * push;
        keepInside(p, bounds[i]);
      }
    }
  }
  state.placed = placed;
  state.placedKey = key;
  return placed;
}

// ---------- setup ----------

export function init(elements) {
  ({ ctx, audio, ui } = elements);

  ui.play.addEventListener("click", togglePlay);
  for (const name of ["play", "pause", "ended"]) audio.addEventListener(name, setPlayButton);

  audio.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(audio.duration) && audio.duration > 0) {
      const lastT = state.stars.length ? state.stars[state.stars.length - 1].t : 0;
      state.duration = Math.ceil(Math.max(audio.duration, lastT));   // whole seconds, so the axis ends on a label
    }
  });

  // The listed preview couldn't be played (deleted, corrupt, unsupported): fall back to scrubbing
  audio.addEventListener("error", () => {
    if (!audio.getAttribute("src") || !state.track || !state.track.preview_url || state.manual) return;
    setManual(true, "The preview couldn't be loaded. Drag to light the stars.");
  });
}

export const currentTrack = () => state.track;

export function layout() {
  const box = ui.plot.getBoundingClientRect();
  const h = Math.max(40, box.height - AXIS_HEIGHT - 10);
  state.plot = { x: box.left, y: box.top + 10, w: box.width, h };
  // Scale stars by the smaller side, so a tall narrow phone plot doesn't get huge overlapping stars
  state.rMax = clamp(Math.min(h / 85, box.width / 150), 2.5, 8);
}

// ---------- time ----------

// audio.currentTime only updates a few times a second in some browsers, so between
// updates the playhead is extrapolated from the wall clock to keep it smooth.
function currentTime(now) {
  if (state.manual) return Number(ui.scrub.value);
  const t = audio.currentTime || 0;
  if (audio.paused || audio.seeking || t !== state.clock.audioT) {
    state.clock = { audioT: t, wall: now };
    return t;
  }
  const ahead = ((now - state.clock.wall) / 1000) * audio.playbackRate;
  return Math.min(t + ahead, t + 0.25, state.duration);
}

export function seekTo(t) {
  t = clamp(t, 0, state.duration);
  if (state.manual) ui.scrub.value = t;
  else if (state.track && audio.readyState > 0) audio.currentTime = t;
}

// Lights every star at or before t; returns the index of the latest lit star (or -1)
function updateLit(t, now) {
  const isSeek = Math.abs(t - state.lastT) > SEEK_JUMP;
  let count = 0;
  let current = -1;
  state.stars.forEach((star, i) => {
    if (star.t <= t) {
      if (state.litAt[i] === null) state.litAt[i] = isSeek || reduceMotion ? -Infinity : now;
      count++;
      current = i;
    } else {
      state.litAt[i] = null;
    }
  });
  state.lastT = t;
  if (count !== state.litCount) {
    state.litCount = count;
    ui.lit.textContent = count;
  }
  return current;
}

// ---------- drawing ----------

function drawSpecks() {
  const { x, y, w, h } = state.plot;
  const alpha = ctx.globalAlpha;
  for (const s of state.specks) {
    ctx.globalAlpha = alpha * s.alpha;
    circle(ctx, x + s.u * w, y + s.v * h, s.r, theme.ink);
  }
  ctx.globalAlpha = alpha;
}

function drawTrend() {
  if (!state.trend || !SKY_SHOW_ARC) return;
  const [c0, c1, c2] = state.trend;
  ctx.save();
  ctx.setLineDash([4, 7]);
  ctx.strokeStyle = ink(0.28 * SKY_ARC_OPACITY);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 80; i++) {
    const u = i / 80;
    const pitch = clamp(c0 + c1 * u + c2 * u * u, state.pitchLo, state.pitchHi);
    const px = state.plot.x + u * state.plot.w;
    if (i === 0) ctx.moveTo(px, yAt(pitch));
    else ctx.lineTo(px, yAt(pitch));
  }
  ctx.stroke();
  ctx.restore();
}

function drawAxis() {
  const { y, h } = state.plot;
  const step = state.duration > 40 ? 10 : 5;
  ctx.font = font(11);
  ctx.fillStyle = theme.muted;
  ctx.textBaseline = "alphabetic";
  for (let s = 0; s <= state.duration + 0.001; s += 1) {
    const px = xAt(s);
    if (s % step === 0) {
      ctx.textAlign = s === 0 ? "left" : s >= state.duration - 0.001 ? "right" : "center";
      ctx.fillText(s === 0 ? "0s" : String(s), px, y + h + 26);
    } else {
      ctx.fillRect(Math.round(px), y + h + 18, 1, 1);    // a faint tick every second
    }
  }
}

function drawStars(now, current) {
  const alpha = ctx.globalAlpha;
  const placed = placedStars();
  state.stars.forEach((star, i) => {
    const { x, y, r } = placed[i];
    const litAt = state.litAt[i];
    const shown = appear(now, state.shownAt, 320, (star.t / state.duration) * INTRO_SWEEP_MS);
    if (shown <= 0) return;
    ctx.globalAlpha = alpha * shown;

    if (litAt === null) {
      if (SKY_TWINKLE > 0 && !reduceMotion) {
        const { twinklePhase, twinklePeriod } = state.scatter[i];
        ctx.globalAlpha = alpha * shown * (1 - SKY_TWINKLE * (0.5 + 0.5 * Math.sin((2 * Math.PI * now) / twinklePeriod + twinklePhase)));
      }
      circle(ctx, x, y, r * (0.4 + 0.6 * shown), theme.starDim);
      return;
    }

    const k = 1 - clamp((now - litAt) / FLASH_MS, 0, 1);   // 1 at the moment it lights, 0 when settled
    if (k > 0) {
      const glowR = r * (2 + 4 * k);
      const glow = ctx.createRadialGradient(x, y, 0, x, y, glowR);
      glow.addColorStop(0, ink(0.6 * k));
      glow.addColorStop(1, ink(0));
      circle(ctx, x, y, glowR, glow);
    }
    ctx.globalAlpha = alpha * shown * state.scatter[i].brightMul;
    circle(ctx, x, y, r * (1 + 0.5 * k * k), theme.ink);
  });
  ctx.globalAlpha = alpha;

  // The latest star: an accent ring and its note name
  if (current >= 0) {
    const star = state.stars[current];
    const { x, y, r } = placed[current];
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, r + 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = font(11, 500);
    ctx.fillStyle = theme.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(noteName(star.pitch), x + r + 12, y + 1);
  }
}

function drawPlayhead(t) {
  const x = Math.round(xAt(t)) + 0.5;
  ctx.save();
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = ink(0.6);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, state.plot.y - 14);
  ctx.lineTo(x, state.plot.y + state.plot.h + 10);
  ctx.stroke();
  ctx.restore();
}

export function draw(now) {
  if (!state.plot || !state.track) return;
  const t = currentTime(now);
  const current = updateLit(t, now);
  const alpha = ctx.globalAlpha;
  ctx.globalAlpha = alpha * appear(now, state.shownAt, INTRO_MS);
  drawSpecks();
  drawTrend();
  drawAxis();
  ctx.globalAlpha = alpha;
  drawStars(now, current);
  drawPlayhead(t);
  const text = `${formatTime(t)} / ${formatTime(state.duration)}`;
  if (text !== state.timeText) ui.time.textContent = state.timeText = text;
}

// ---------- songs ----------

function setManual(on, message) {
  state.manual = on;
  ui.manual.hidden = !on;
  ui.play.hidden = on;
  ui.scrub.max = state.duration;
  ui.scrub.value = 0;
  if (message) ui.hint.textContent = message;
  layout();
}

function setPlayButton() {
  const playing = !audio.paused && !audio.ended;
  ui.play.querySelector(".icon").textContent = playing ? "❚❚" : "▶";
  ui.play.querySelector(".label").textContent = playing ? "Pause" : "Play";
}

function togglePlay() {
  if (audio.paused || audio.ended) audio.play().catch((error) => console.warn("Playback failed", error));
  else audio.pause();
}

export function stop() {
  audio.pause();
}

// Shows a song: fills the header and footer, fetches its star map and points the audio at its preview
export async function load(track) {
  const token = ++state.loadToken;
  audio.pause();
  audio.removeAttribute("src");
  audio.load();

  state.track = track;
  state.stars = [];                                // don't show the previous song's sky while loading
  state.litAt = [];
  state.scatter = [];
  state.placedKey = "";
  state.trend = null;
  state.shownAt = Infinity;
  replayEntrance(ui.titles);
  ui.song.value = String(track.deezer_id);
  ui.title.textContent = track.title;
  ui.artist.textContent = track.artist;
  ui.artist.title = track.artist;
  ui.genre.textContent = track.genre || "—";
  ui.tempo.textContent = track.tempo ? `${track.tempo} BPM` : "—";
  ui.total.textContent = "…";
  ui.counter.classList.add("is-loading");

  let stars = [];
  try {
    const response = await fetch(`/api/stars/${track.deezer_id}`);
    if (response.ok) stars = await response.json();
  } catch (error) {
    console.warn("Could not load the star map", error);
  }
  if (token !== state.loadToken) return;          // another song was picked meanwhile

  stars.sort((a, b) => a.t - b.t);
  const pitches = stars.map((s) => s.pitch);
  let lo = pitches.length ? Math.min(...pitches) - 2 : 36;
  let hi = pitches.length ? Math.max(...pitches) + 2 : 84;
  if (hi - lo < MIN_PITCH_SPAN) {
    const mid = (lo + hi) / 2;
    lo = mid - MIN_PITCH_SPAN / 2;
    hi = mid + MIN_PITCH_SPAN / 2;
  }
  const lastT = stars.length ? stars[stars.length - 1].t : 0;

  const random = seededRandom(track.deezer_id);
  Object.assign(state, {
    stars,
    litAt: stars.map(() => null),
    scatter: scatterStars(stars, track.deezer_id),
    placedKey: "",
    pitchLo: lo,
    pitchHi: hi,
    duration: Math.max(PREVIEW_SECONDS, Math.ceil(lastT)),
    lastT: 0,
    litCount: -1,
    specks: Array.from({ length: SPECKS }, () => ({
      u: random(), v: random(), r: 0.4 + random() * 0.7, alpha: 0.08 + random() * 0.22,
    })),
  });
  state.trend = quadraticTrend(stars, state.duration);
  state.shownAt = performance.now();
  ui.total.textContent = stars.length;
  ui.counter.classList.remove("is-loading");

  if (track.preview_url) {
    setManual(false);
    ui.play.disabled = false;
    audio.src = track.preview_url;
    audio.load();
  } else {
    setManual(true, "No preview for this song. Drag to light the stars.");
  }
  setPlayButton();
}

// ---------- input ----------

const inPlot = (e) =>
  state.plot &&
  e.clientX >= state.plot.x && e.clientX <= state.plot.x + state.plot.w &&
  e.clientY >= state.plot.y - 14 && e.clientY <= state.plot.y + state.plot.h + AXIS_HEIGHT;

export function pointerDown(e, canvas) {
  if (!inPlot(e)) return;
  state.dragging = true;
  canvas.setPointerCapture(e.pointerId);
  seekTo(timeAtX(e.clientX));
}

export function pointerMove(e, canvas) {
  if (state.dragging) seekTo(timeAtX(e.clientX));
  else canvas.style.cursor = inPlot(e) ? "pointer" : "default";
}

export function pointerUp() {
  state.dragging = false;
}

// Space plays or pauses, arrows skip 2 s. Returns true if the key was used.
export function keydown(e) {
  if (e.key === " " && !state.manual && state.track) {
    togglePlay();
    return true;
  }
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    seekTo(currentTime(performance.now()) + (e.key === "ArrowRight" ? 2 : -2));
    return true;
  }
  return false;
}
