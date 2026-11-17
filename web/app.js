// Sonata's web UI: the galaxy (home, "/") and a song's sky ("/?id=<deezer_id>"),
// drawn on one shared canvas. Clicking a star dives into it and fades into its sky;
// the back button (or Escape, or the browser's back) pulls back out to the galaxy.
import * as galaxy from "./galaxy.js";
import * as panels from "./panels.js";
import * as sky from "./sky.js";
import { clamp, loadTheme } from "./util.js";

const DIVE_MS = 650;           // galaxy zooms into the star
const SKY_FADE_MS = 450;
const RETURN_MS = 750;         // galaxy zooms back out

const $ = (id) => document.getElementById(id);
const canvas = $("stage");
const ctx = canvas.getContext("2d");
const audio = $("audio");
const views = { galaxy: $("galaxy-view"), sky: $("sky-view") };
const LABELS = {
  galaxy: "Galaxy of your library: one star per song, songs that sound alike sit close together, liked songs are bright",
  sky: "Star map of the song: each star is a note onset, placed left to right by time and bottom to top by pitch, sized by loudness",
};

let library = [];
let active = "galaxy";         // the view that gets input
let busy = false;              // during a transition, input is ignored

// Each view's canvas opacity fades linearly between values
const fades = { galaxy: { from: 1, to: 1, start: 0, duration: 1 }, sky: { from: 0, to: 0, start: 0, duration: 1 } };

function opacity(name, now) {
  const f = fades[name];
  return f.from + (f.to - f.from) * clamp((now - f.start) / f.duration, 0, 1);
}

function fadeTo(name, to, duration, from) {
  const now = performance.now();
  fades[name] = { from: from ?? opacity(name, now), to, start: now, duration };
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const findTrack = (id) => library.find((t) => String(t.deezer_id) === String(id));
const idFromUrl = () => new URLSearchParams(location.search).get("id");

function showView(name) {
  active = name;
  for (const [key, element] of Object.entries(views)) {
    element.classList.toggle("is-active", key === name);
    element.classList.remove("is-leaving");
    element.inert = key !== name;
  }
  canvas.setAttribute("aria-label", LABELS[name]);
  canvas.style.cursor = name === "galaxy" ? "grab" : "default";
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  galaxy.layout();
  sky.layout();
}

function frame(now) {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const g = opacity("galaxy", now);
  const s = opacity("sky", now);
  if (g > 0.001) {
    ctx.globalAlpha = g;
    galaxy.draw(now);
  }
  if (s > 0.001) {
    ctx.globalAlpha = s;
    sky.draw(now);
  }
  ctx.globalAlpha = 1;
  requestAnimationFrame(frame);
}

// ---------- navigation ----------

async function openSong(track, { push = true, animate = true } = {}) {
  if (push) history.pushState({ id: track.deezer_id, fromGalaxy: true }, "", `?id=${track.deezer_id}`);
  document.title = `${track.title} · Sonata`;
  const loading = sky.load(track);

  setSheet(false);
  if (animate && active === "galaxy" && Number.isFinite(track.x)) {
    busy = true;
    views.galaxy.classList.add("is-leaving");
    galaxy.zoomInto(track, DIVE_MS);
    fadeTo("galaxy", 0, DIVE_MS);
    await Promise.all([loading, wait(DIVE_MS * 0.75)]);
  }
  showView("sky");
  fadeTo("galaxy", 0, 200);
  fadeTo("sky", 1, animate ? SKY_FADE_MS : 1, animate ? 0 : 1);
  busy = false;
}

function showGalaxy({ animate = true } = {}) {
  const track = sky.currentTrack();
  sky.stop();
  document.title = "Sonata";
  showView("galaxy");
  fadeTo("sky", 0, animate ? 250 : 1);
  if (animate) {
    galaxy.zoomOutFrom(track, RETURN_MS);
    fadeTo("galaxy", 1, RETURN_MS * 0.8, 0);
  } else {
    fadeTo("galaxy", 1, 1);
  }
}

function goBack() {
  if (history.state?.fromGalaxy) {
    history.back();                       // popstate below does the transition
  } else {
    history.pushState({}, "", location.pathname);
    showGalaxy();
  }
}

window.addEventListener("popstate", () => {
  const track = findTrack(idFromUrl());
  if (track) openSong(track, { push: false, animate: active === "galaxy" });
  else if (active === "sky") showGalaxy();
});

// ---------- input ----------

const activeView = () => (active === "galaxy" ? galaxy : sky);

canvas.addEventListener("pointerdown", (e) => {
  if (setSheet(false)) return;           // first tap outside an open sheet just closes it
  if (!busy) activeView().pointerDown(e, canvas);
});
canvas.addEventListener("pointermove", (e) => { if (!busy) activeView().pointerMove(e, canvas); });
canvas.addEventListener("pointerup", (e) => activeView().pointerUp(e, canvas));
canvas.addEventListener("pointercancel", (e) => activeView().pointerUp(e, canvas));
canvas.addEventListener("pointerleave", () => { if (active === "galaxy") galaxy.pointerLeave(); });
canvas.addEventListener("wheel", (e) => { if (active === "galaxy" && !busy) galaxy.wheel(e); }, { passive: false });

document.addEventListener("keydown", (e) => {
  if (busy || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest("select, input")) return;
  if (e.key === "Escape" && setSheet(false)) return;
  if (active === "sky" && e.key === "Escape") {
    e.preventDefault();
    goBack();
    return;
  }
  if (e.target.closest("button") && (e.key === " " || e.key === "Enter")) return;
  if (activeView().keydown(e)) e.preventDefault();
});

$("back").addEventListener("click", goBack);

// Phones: "For you" opens as a bottom sheet. Returns true if it was open (and is now closed).
const sheet = $("for-you");
const sheetToggle = $("for-you-toggle");
function setSheet(open) {
  const wasOpen = sheet.classList.contains("is-open");
  sheet.classList.toggle("is-open", open);
  sheetToggle.setAttribute("aria-expanded", String(open));
  if (open) sheet.querySelector("button, [tabindex]")?.focus({ preventScroll: true });
  else if (wasOpen && sheet.contains(document.activeElement)) sheetToggle.focus({ preventScroll: true });
  return wasOpen && !open;
}
sheetToggle.addEventListener("click", () => setSheet(!sheet.classList.contains("is-open")));
$("for-you-close").addEventListener("click", () => setSheet(false));

// The sky's song picker swaps songs in place; the galaxy's dives into the chosen star
$("song").addEventListener("change", (e) => {
  const track = findTrack(e.target.value);
  if (!track) return;
  history.replaceState({ ...history.state, id: track.deezer_id }, "", `?id=${track.deezer_id}`);
  openSong(track, { push: false, animate: false });
});
$("g-find").addEventListener("change", (e) => {
  const track = findTrack(e.target.value);
  e.target.value = "";
  if (track) openSong(track);
});

function fillPickers() {
  for (const select of [$("song"), $("g-find")]) {
    for (const [label, liked] of [["Your songs ♥", true], ["Discover", false]]) {
      const tracks = library.filter((t) => Boolean(t.liked) === liked);
      if (!tracks.length) continue;
      const group = document.createElement("optgroup");
      group.label = label;
      for (const track of tracks) {
        const option = document.createElement("option");
        option.value = String(track.deezer_id);
        option.textContent = `${track.title} — ${track.artist}${track.preview_url ? "" : "  (no preview)"}`;
        group.append(option);
      }
      select.append(group);
    }
  }
}

// ---------- start ----------

async function start() {
  loadTheme();                           // colors and font for the canvas, from style.css
  sky.init({
    ctx, audio,
    ui: {
      title: $("title"), lit: $("lit"), total: $("total"), song: $("song"), play: $("play"),
      plot: $("plot"), manual: $("manual"), scrub: $("scrub"), hint: $("hint"),
      artist: $("artist"), genre: $("genre"), tempo: $("tempo"), time: $("time"),
      titles: $("sky-titles"), counter: $("sky-counter"),
    },
  });
  galaxy.init({
    ctx,
    onOpen: (track) => openSong(track),
    ui: {
      frame: views.galaxy, area: $("galaxy-area"), recenter: $("recenter"), hint: $("g-hint"),
      liked: $("g-liked"), total: $("g-total"),
      tooltip: $("tooltip"), tipTitle: $("tip-title"), tipMeta: $("tip-meta"), askBox: $("ask"), askForm: $("ask-form"),
    },
  });

  // A song link starts in the sky straight away: no galaxy flash, no fade on page load
  if (idFromUrl()) {
    showView("sky");
    fades.galaxy = { from: 0, to: 0, start: 0, duration: 1 };
    fades.sky = { from: 1, to: 1, start: 0, duration: 1 };
  }
  requestAnimationFrame(() => requestAnimationFrame(() => document.documentElement.classList.remove("instant")));

  const observer = new ResizeObserver(() => resize());
  for (const element of [$("plot"), $("galaxy-area")]) observer.observe(element);
  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(frame);
  panels.showRecommendationsLoading($("rec-list"));

  try {
    const response = await fetch("/api/library");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    library = await response.json();
  } catch (error) {
    $("galaxy-loading").classList.remove("is-working");
    $("galaxy-loading").textContent = "Couldn't load the library. Is python -m sonata.server running?";
    $("rec-list").replaceChildren();
    console.error(error);
    return;
  }
  $("galaxy-loading").classList.add("is-done");
  $("g-counter").classList.remove("is-loading");
  fillPickers();
  galaxy.setLibrary(library);

  // "For you" (content-based) and Ask Sonata (RAG); both open a song by diving into its star
  const lists = {
    findTrack,
    onOpen: (track) => { galaxy.highlight(null); openSong(track); },
    onHighlight: (id) => galaxy.highlight(id),
  };
  panels.loadRecommendations({ ...lists, list: $("rec-list"), onLoaded: (recs) => galaxy.setRecommendations(recs) });
  panels.initAsk({
    ...lists,
    form: $("ask-form"), input: $("ask-input"), button: $("ask-button"),
    answers: $("answers"), status: $("answers-status"), cards: $("answer-cards"), clear: $("answers-clear"),
    onAnswers: (ids) => galaxy.setAnswers(ids),
  });

  // A shared song link opens straight into its sky; back then goes to the galaxy
  const track = findTrack(idFromUrl());
  if (track) {
    history.replaceState({ id: track.deezer_id, fromGalaxy: false }, "", `?id=${track.deezer_id}`);
    openSong(track, { push: false, animate: false });
  } else if (idFromUrl()) {
    history.replaceState({}, "", location.pathname);   // unknown song: fall back to the galaxy
    showGalaxy({ animate: false });
  }
}

start();
