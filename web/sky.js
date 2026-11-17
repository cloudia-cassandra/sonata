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
  if (!state.trend) return;
  const [c0, c1, c2] = state.trend;
  ctx.save();
  ctx.setLineDash([4, 7]);
  ctx.strokeStyle = ink(0.28);
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
  state.stars.forEach((star, i) => {
    const x = xAt(star.t);
    const y = yAt(star.pitch);
    const r = radiusOf(star.size);
    const litAt = state.litAt[i];
    const shown = appear(now, state.shownAt, 320, (star.t / state.duration) * INTRO_SWEEP_MS);
    if (shown <= 0) return;
    ctx.globalAlpha = alpha * shown;

    if (litAt === null) {
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
    circle(ctx, x, y, r * (1 + 0.5 * k * k), theme.ink);
  });
  ctx.globalAlpha = alpha;

  // The latest star: an accent ring and its note name
  if (current >= 0) {
    const star = state.stars[current];
    const x = xAt(star.t);
    const y = yAt(star.pitch);
    const r = radiusOf(star.size);
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
