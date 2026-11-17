// Audio effects for the song sky: speed (pitch follows, like "slowed" and "sped up" edits),
// reverb and 8D panning, in a slim panel at the bottom of the sky.
//
// Self-contained: it finds the sky's <audio> and Play button by id and adds its own panel
// and styles. Set FX_ENABLED to false and it does nothing at all.
//
// Safety:
// - The default settings (speed 1, reverb 0, 8D off) are neutral, and at them the audio
//   isn't routed through Web Audio at all: it plays exactly as before. The <audio> element
//   joins the effects graph only when reverb or 8D is first turned on (and only while the
//   AudioContext is running, so it can never be routed into a silent, suspended context).
//   That connection is permanent, as createMediaElementSource allows only one per element;
//   with neutral settings the graph then passes the sound through unchanged.
// - Speed is audio.playbackRate. The sky's stars keep following audio.currentTime, which is
//   in song time at any speed, so they stay in sync.
// - Any failure is logged, the panel is removed and playback carries on plain.
const FX_ENABLED = true;

const SPEED_MIN = 0.5;
const SPEED_MAX = 1.5;
const REVERB_SECONDS = 2.5;    // length of the generated impulse response
const WET_LEVEL = 0.9;         // reverb return at 100%
const DRY_DUCK = 0.5;          // the dry signal drops by up to this much as reverb rises
const PAN_PERIOD_S = 8;        // 8D: one full left-right-left sweep
const RAMP_S = 0.05;           // time constant for smooth changes to gains
const PAN_RAMP_S = 0.4;        // ...and for 8D fading in and out
const IDLE_MS = 4000;          // the panel fades when the mouse hasn't moved for this long

const NEUTRAL = { speed: 1, reverb: 0, eightD: false };
const PRESETS = [
  { name: "Normal", ...NEUTRAL },
  { name: "Slowed + Reverb", speed: 0.85, reverb: 0.45, eightD: false },
  { name: "Sped up", speed: 1.25, reverb: 0.1, eightD: false },
  { name: "8D", speed: 1, reverb: 0.2, eightD: true },
];

// Module state, so settings carry over from song to song (the sky reuses one <audio> element)
const settings = { ...NEUTRAL };
let audio, playButton, panel, style, controls;
let ctx = null;                // the AudioContext, created on the first Play
let graph = null;              // the effects graph, built when first needed
let failed = false;
let idleTimer = 0;

// ---------- failure ----------

function fail(where, error) {
  if (failed) return;
  failed = true;
  console.error(`Sonata effects: ${where} failed, so playback continues without effects.`, error);
  panel?.remove();
  style?.remove();
  document.documentElement.removeAttribute("data-fx-reverb");
  try {
    if (audio) {
      audio.defaultPlaybackRate = 1;
      audio.playbackRate = 1;
      setPreservesPitch(true);
    }
    // If the element was already routed through the graph, wire it straight to the speakers
    if (graph) {
      graph.source.disconnect();
      graph.source.connect(ctx.destination);
      if (ctx.state !== "running") ctx.resume().catch(() => {});
    }
  } catch (cleanupError) {
    console.error("Sonata effects: cleanup failed", cleanupError);
  }
}

// Every event handler runs through this, so an error can never escape into the player
const guard = (where, fn) => (...args) => {
  if (failed) return;
  try {
    fn(...args);
  } catch (error) {
    fail(where, error);
  }
};

// ---------- speed ----------

function setPreservesPitch(on) {
  audio.preservesPitch = on;
  audio.mozPreservesPitch = on;
  audio.webkitPreservesPitch = on;
}

// Loading a new song resets playbackRate to defaultPlaybackRate, so both are set
function applySpeed() {
  setPreservesPitch(settings.speed === 1);   // off: slower is lower, faster is higher
  audio.defaultPlaybackRate = settings.speed;
  if (audio.playbackRate !== settings.speed) audio.playbackRate = settings.speed;
}

// ---------- the Web Audio graph ----------

const needsGraph = () => settings.reverb > 0 || settings.eightD;

// Called from the Play button (a user action, which browsers require before audio can start)
function ensureContext() {
  if (!ctx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    ctx = new AudioContextClass();
    ctx.addEventListener("statechange", guard("audio context", connectIfNeeded));
  }
  if (ctx.state !== "running") ctx.resume().then(guard("audio resume", connectIfNeeded), (error) => console.warn("Sonata effects: audio is suspended", error));
  connectIfNeeded();
}

function connectIfNeeded() {
  if (failed || graph || !ctx || ctx.state !== "running" || !needsGraph()) return;
  // A cross-origin preview without CORS would come out of the graph silent; leave it plain
  const src = audio.currentSrc || audio.src;
  if (src && new URL(src, location.href).origin !== location.origin) return;
  graph = buildGraph();
  updateGraph(true);
}

//  <audio> ─┬─ dry gain ───────────────┬─ stereo panner ── speakers
//           └─ convolver ── wet gain ──┘        ↑ pan
//                         sine LFO ── depth gain ┘
function buildGraph() {
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  wet.gain.value = 0;
  const convolver = ctx.createConvolver();
  convolver.buffer = impulseResponse(REVERB_SECONDS);

  const panner = ctx.createStereoPanner();
  // Treat input as stereo: a mono preview is upmixed to both ears at full level, as plain
  // <audio> plays it, instead of being equal-power panned 3 dB quieter
  panner.channelCount = 2;
  panner.channelCountMode = "explicit";

  // 8D: a slow sine moves the pan from -1 to 1 and back; depth 0 holds it in the center
  const lfo = ctx.createOscillator();
  lfo.frequency.value = 1 / PAN_PERIOD_S;
  const depth = ctx.createGain();
  depth.gain.value = 0;
  lfo.connect(depth).connect(panner.pan);
  lfo.start();

  dry.connect(panner);
  convolver.connect(wet).connect(panner);
  panner.connect(ctx.destination);

  // Last, once everything after it is in place. Only ever once per element.
  const source = ctx.createMediaElementSource(audio);
  source.connect(dry);
  source.connect(convolver);
  return { source, dry, wet, depth };
}

// A room's echo, made in code: stereo noise, gently low-passed for a softer tail,
// decaying to -60 dB over the length
function impulseResponse(seconds) {
  const rate = ctx.sampleRate;
  const length = Math.round(rate * seconds);
  const buffer = ctx.createBuffer(2, length, rate);
  const decay = Math.log(1000) / seconds;
  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    let smoothed = 0;
    for (let i = 0; i < length; i++) {
      const t = i / rate;
      smoothed += 0.35 * (Math.random() * 2 - 1 - smoothed);
      data[i] = smoothed * Math.exp(-decay * t) * Math.min(1, t / 0.008);   // 8 ms fade-in, no click
    }
  }
  return buffer;
}

function updateGraph(immediate = false) {
  if (!graph) return;
  const now = ctx.currentTime;
  const set = (param, value, tau = RAMP_S) => {
    param.cancelScheduledValues(now);
    if (immediate) param.setValueAtTime(value, now);
    else param.setTargetAtTime(value, now, tau);
  };
  set(graph.dry.gain, 1 - DRY_DUCK * settings.reverb);
  set(graph.wet.gain, WET_LEVEL * settings.reverb);
  set(graph.depth.gain, settings.eightD ? 1 : 0, PAN_RAMP_S);
}

// ---------- applying settings ----------

function apply() {
  applySpeed();
  if (ctx) {
    // A panel click is a user action too, so a suspended context may be resumed here
    if (ctx.state !== "running" && needsGraph()) ctx.resume().then(guard("audio resume", connectIfNeeded), () => {});
    connectIfNeeded();
    updateGraph();
  }
  document.documentElement.dataset.fxReverb = String(settings.reverb);   // read by the sky for lingering stars
  syncPanel();
}

function set(changes) {
  Object.assign(settings, changes);
  settings.speed = Math.min(SPEED_MAX, Math.max(SPEED_MIN, settings.speed));
  settings.reverb = Math.min(1, Math.max(0, settings.reverb));
  apply();
}

// ---------- panel ----------

const CSS = `
.fx {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  pointer-events: auto;
  transition: opacity 0.6s var(--ease);
}
.fx[hidden] { display: none; }
.fx.is-idle:not(:hover):not(:focus-within) { opacity: 0.22; }
.fx-row { display: flex; align-items: center; gap: 8px 10px; min-width: 0; }
.fx-presets { overflow-x: auto; scrollbar-width: none; }
.fx-presets::-webkit-scrollbar { display: none; }
.fx-name { flex: none; width: 64px; color: var(--muted); }
.fx button { flex: none; height: 28px; padding: 0 11px; }
.fx button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.fx .text-button { height: 28px; padding: 0 2px; }
.fx-controls { flex-wrap: wrap; gap: 6px 24px; }
.fx-slider { display: flex; flex: 0 1 260px; align-items: center; gap: 10px; min-width: 180px; }
.fx-slider .label { flex: none; color: var(--muted); }
.fx-slider input { flex: 1; min-width: 60px; }
.fx-slider output { flex: none; width: 5ch; color: var(--ink-soft); font-size: var(--text-small); text-align: right; }
.fx-buttons { display: flex; align-items: center; gap: 14px; }
@media (max-width: 720px) {
  .fx { gap: 6px; padding-top: 8px; }
  .fx-name { display: none; }
  .fx-presets { flex-wrap: wrap; gap: 6px; }
  .fx-presets button { padding: 0 9px; letter-spacing: 0.1em; }
  .fx-controls { gap: 4px 16px; }
  .fx-slider { flex: 1 1 100%; min-width: 0; }
}
`;

// Properties are set directly; "role" and "aria-*" keys become attributes
function element(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "role" || key.startsWith("aria-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  node.append(...children);
  return node;
}

function slider(label, min, max, step) {
  const input = element("input", { type: "range", min, max, step });
  const output = element("output");
  const wrapper = element("label", { className: "fx-slider" }, [element("span", { className: "label", textContent: label }), input, output]);
  return { wrapper, input, output };
}

function buildPanel() {
  style = element("style", { id: "fx-style", textContent: CSS });
  document.head.append(style);

  const presetButtons = PRESETS.map((preset) => {
    const button = element("button", { type: "button", textContent: preset.name });
    button.addEventListener("click", guard("preset", () => set({ speed: preset.speed, reverb: preset.reverb, eightD: preset.eightD })));
    return button;
  });
  const speed = slider("Speed", SPEED_MIN, SPEED_MAX, 0.05);
  const reverb = slider("Reverb", 0, 100, 1);
  const eightD = element("button", { type: "button", textContent: "8D", title: "Sound circles slowly around your head" });
  const reset = element("button", { type: "button", className: "text-button", textContent: "Reset" });

  speed.input.addEventListener("input", guard("speed", () => set({ speed: Number(speed.input.value) })));
  reverb.input.addEventListener("input", guard("reverb", () => set({ reverb: Number(reverb.input.value) / 100 })));
  eightD.addEventListener("click", guard("8D", () => set({ eightD: !settings.eightD })));
  reset.addEventListener("click", guard("reset", () => set({ ...NEUTRAL })));

  panel = element("section", { className: "fx", role: "group", "aria-label": "Audio effects" }, [
    element("div", { className: "fx-row fx-presets", role: "group", "aria-label": "Presets" },
      [element("span", { className: "label fx-name", textContent: "Effects" }), ...presetButtons]),
    element("div", { className: "fx-row fx-controls" },
      [speed.wrapper, reverb.wrapper, element("div", { className: "fx-buttons" }, [eightD, reset])]),
  ]);
  controls = { presetButtons, speed, reverb, eightD };

  // Between the sky and its footer, so the sky's plot shrinks to make room
  const skyView = document.getElementById("sky-view");
  skyView.insertBefore(panel, skyView.querySelector(".foot"));
  syncPanel();
}

function syncPanel() {
  if (!panel) return;
  const { presetButtons, speed, reverb, eightD } = controls;
  speed.input.value = String(settings.speed);
  speed.output.value = `${settings.speed.toFixed(2)}×`;
  reverb.input.value = String(Math.round(settings.reverb * 100));
  reverb.output.value = `${Math.round(settings.reverb * 100)}%`;
  eightD.setAttribute("aria-pressed", String(settings.eightD));
  PRESETS.forEach((preset, i) => {
    const on = Math.abs(preset.speed - settings.speed) < 1e-6 && Math.abs(preset.reverb - settings.reverb) < 1e-6 && preset.eightD === settings.eightD;
    presetButtons[i].setAttribute("aria-pressed", String(on));
  });
}

// Fades after IDLE_MS without the mouse moving; any movement, touch or key brings it back
function wake() {
  panel?.classList.remove("is-idle");
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => panel?.classList.add("is-idle"), IDLE_MS);
}

// ---------- start ----------

function init() {
  if (!FX_ENABLED) return;
  try {
    audio = document.getElementById("audio");
    playButton = document.getElementById("play");
    if (!audio || !playButton || !document.getElementById("sky-view")) throw new Error("the song sky's elements are missing");
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || typeof AudioContextClass.prototype.createStereoPanner !== "function") {
      throw new Error("this browser lacks the Web Audio features the effects need");
    }

    buildPanel();

    // The context is created on the first Play; the "play" event also catches the Space key
    playButton.addEventListener("click", guard("audio setup", ensureContext));
    audio.addEventListener("play", guard("audio setup", ensureContext));
    // A new song's load() resets playbackRate, so the speed is put back
    audio.addEventListener("loadstart", guard("speed", applySpeed));

    // Songs without a preview are scrubbed by hand and have no sound: no panel then
    const syncVisible = guard("panel", () => { panel.hidden = playButton.hidden; });
    new MutationObserver(syncVisible).observe(playButton, { attributes: true, attributeFilter: ["hidden"] });
    syncVisible();

    for (const name of ["pointermove", "pointerdown", "keydown", "wheel"]) {
      window.addEventListener(name, wake, { passive: true });
    }
    wake();
    apply();
  } catch (error) {
    fail("setup", error);
  }
}

init();
