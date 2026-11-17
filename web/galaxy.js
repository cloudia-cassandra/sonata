// The galaxy: one star per song, placed by sonata.compute_galaxy (UMAP of the audio
// features), so songs that sound alike sit close together. Liked songs shine, the rest
// are dim. Drag to pan, scroll or pinch to zoom, click a star to open its sky.
import { appear, circle, clamp, easeInOut, font, reduceMotion, seededRandom, theme } from "./util.js";

const MIN_SCALE = 0.6;
const MAX_SCALE = 40;
const HIT_PX = 12;             // how close the pointer must be to pick a star
const CLICK_SLOP = 5;          // px of movement that still counts as a click, not a drag
const LABEL_FROM_SCALE = 3.5;  // show song titles next to stars from this zoom on
const MAX_LABELS = 80;
const SPECKS = 260;
const PARALLAX = 0.12;         // background specks drift slower than the stars
const ZOOM_IN = 7;             // how far the camera dives into a star when opening it

let ctx, ui, onOpen;

const g = {
  stars: [],                   // {track, x, y, phase, sx, sy}; liked last so they draw on top
  extent: { x: 1, y: 1 },
  frame: null,                 // the bordered frame's inner rect: stars are clipped to it
  area: null,                  // the open space between header and footer: labels stay inside it
  cx: 0, cy: 0, base: 1,       // screen center of the galaxy area and px per world unit at scale 1
  cam: { x: 0, y: 0, scale: 1 },
  saved: null,                 // the camera before diving into a song, to return to
  tween: null,
  hover: -1,
  tipWidth: 0,
  pointers: new Map(),
  gesture: null,
  specks: [],
  glow: null,
  accentGlow: null,
  index: new Map(),            // deezer_id -> position in stars
  recs: new Map(),             // recommended: deezer_id -> {rank, closestId}
  answers: new Map(),          // Ask Sonata answers: deezer_id -> rank
  recsAt: 0,                   // when recommendations / answers arrived (their entrance animations)
  answersAt: 0,
  dimFrom: 1, dimTo: 1, dimChangedAt: 0,   // other stars dim while answers are shown
};

const ANSWER_DIM = 0.4;        // other stars fade to this while answers are shown

const home = () => ({ x: 0, y: 0, scale: 1 });

function toScreen(x, y) {
  const k = g.base * g.cam.scale;
  return [g.cx + (x - g.cam.x) * k, g.cy - (y - g.cam.y) * k];
}

function toWorld(sx, sy) {
  const k = g.base * g.cam.scale;
  return [g.cam.x + (sx - g.cx) / k, g.cam.y - (sy - g.cy) / k];
}

function makeGlow(rgb) {
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = 64;
  const c = sprite.getContext("2d");
  const gradient = c.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, `rgba(${rgb}, 0.9)`);
  gradient.addColorStop(0.25, `rgba(${rgb}, 0.22)`);
  gradient.addColorStop(1, `rgba(${rgb}, 0)`);
  c.fillStyle = gradient;
  c.fillRect(0, 0, 64, 64);
  return sprite;
}

// ---------- setup ----------

export function init(elements) {
  ({ ctx, ui, onOpen } = elements);
  g.glow = makeGlow(theme.inkRgb);
  g.accentGlow = makeGlow(theme.accentRgb);
  ui.recenter.addEventListener("click", recenter);
}

export function setLibrary(library) {
  g.stars = library
    .filter((t) => Number.isFinite(t.x) && Number.isFinite(t.y))
    .map((t) => ({ track: t, x: t.x, y: t.y, phase: ((t.deezer_id % 997) / 997) * Math.PI * 2, sx: NaN, sy: NaN }))
    .sort((a, b) => Number(a.track.liked) - Number(b.track.liked));
  g.index = new Map(g.stars.map((s, i) => [s.track.deezer_id, i]));
  g.extent = {
    x: Math.max(1e-6, ...g.stars.map((s) => Math.abs(s.x))),
    y: Math.max(1e-6, ...g.stars.map((s) => Math.abs(s.y))),
  };
  ui.liked.textContent = g.stars.filter((s) => s.track.liked).length;
  ui.total.textContent = g.stars.length;
  if (!g.stars.length) {
    ui.hint.textContent = library.length
      ? "No galaxy yet. Run python -m sonata.compute_galaxy to place your songs."
      : "No songs yet.";
  }
  const random = seededRandom(7);
  g.specks = Array.from({ length: SPECKS }, () => ({
    u: random(), v: random(), r: 0.3 + random() * 0.8, alpha: 0.05 + random() * 0.2,
  }));
  layout();
}

export function layout() {
  const area = ui.area.getBoundingClientRect();
  const frame = ui.frame.getBoundingClientRect();
  g.frame = { x: frame.left + 1, y: frame.top + 1, w: frame.width - 2, h: frame.height - 2 };
  g.area = { x: area.left, y: area.top, w: area.width, h: area.height };
  // The Ask box floats over the bottom of the area: fit the galaxy into the space above it
  const occluded = ui.askForm.offsetHeight + 24;
  const freeH = Math.max(60, area.height - occluded);
  g.cx = area.left + area.width / 2;
  g.cy = area.top + freeH / 2;
  g.base = Math.max(1, Math.min(area.width / (2 * g.extent.x), freeH / (2 * g.extent.y)) * 0.94);
}

// ---------- camera ----------

// Moves the camera so the target's on-screen offset shrinks linearly while the zoom
// changes geometrically: the target glides straight to the center instead of swinging out.
function tweenTo(to, duration, now = performance.now()) {
  g.tween = { from: { ...g.cam }, to, start: now, duration };
}

function stepTween(now) {
  if (!g.tween) return;
  const { from, to, start, duration } = g.tween;
  const p = clamp((now - start) / duration, 0, 1);
  const e = easeInOut(p);
  const scale = Math.exp(Math.log(from.scale) + (Math.log(to.scale) - Math.log(from.scale)) * e);
  g.cam = {
    x: to.x - ((1 - e) * (to.x - from.x) * from.scale) / scale,
    y: to.y - ((1 - e) * (to.y - from.y) * from.scale) / scale,
    scale,
  };
  if (p >= 1) g.tween = null;
}

function zoomAt(sx, sy, factor) {
  const [wx, wy] = toWorld(sx, sy);
  const scale = clamp(g.cam.scale * factor, MIN_SCALE, MAX_SCALE);
  const k = g.base * scale;
  g.cam = { x: wx - (sx - g.cx) / k, y: wy + (sy - g.cy) / k, scale };
}

function pan(dx, dy) {
  const k = g.base * g.cam.scale;
  g.cam = { ...g.cam, x: g.cam.x - dx / k, y: g.cam.y + dy / k };
}

export function recenter() {
  tweenTo(home(), 600);
}

// Dive into a song's star (opening its sky)
export function zoomInto(track, duration) {
  clearHover();
  g.saved = { ...g.cam };
  tweenTo({ x: track.x, y: track.y, scale: g.cam.scale * ZOOM_IN }, duration);
}

// Pull back out of a song's star to where the galaxy was (or the home view)
export function zoomOutFrom(track, duration) {
  const back = g.saved || home();
  g.saved = null;
  if (Number.isFinite(track?.x)) {
    g.cam = { x: track.x, y: track.y, scale: back.scale * ZOOM_IN };
    tweenTo(back, duration);
  } else {
    g.cam = back;
  }
}

// ---------- drawing ----------

function drawSpecks(alpha) {
  const { x, y, w, h } = g.frame;
  // Specks drift with the camera at a fraction of the speed, wrapping around the frame
  const k = g.base * PARALLAX;
  const ox = (((-g.cam.x * k) % w) + w) % w;
  const oy = (((g.cam.y * k) % h) + h) % h;
  for (const s of g.specks) {
    ctx.globalAlpha = alpha * s.alpha;
    circle(ctx, x + ((s.u * w + ox) % w), y + ((s.v * h + oy) % h), s.r, theme.ink);
  }
}

// Song titles next to stars when zoomed in: only inside the open galaxy area (not under
// the header or footer text) and never on top of another label. Liked songs go first.
function drawLabels(alpha) {
  ctx.font = font(10);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = theme.muted;
  const { x: ax, y: ay, w: aw, h: ah } = g.area;
  const placed = [];
  for (let i = g.stars.length - 1; i >= 0 && placed.length < MAX_LABELS; i--) {
    const s = g.stars[i];
    if (!Number.isFinite(s.sx) || i === g.hover || kindOf(s) !== "plain") continue;   // those get rank numbers
    const box = { x: s.sx + 8, y: s.sy - 7, w: ctx.measureText(s.track.title).width, h: 14 };
    const inside = box.x >= ax && box.y >= ay && box.x + box.w <= ax + aw && box.y + box.h <= ay + ah;
    if (!inside) continue;
    if (placed.some((p) => box.x < p.x + p.w + 6 && p.x < box.x + box.w + 6 && box.y < p.y + p.h && p.y < box.y + box.h)) continue;
    placed.push(box);
    ctx.globalAlpha = alpha * (s.track.liked ? 0.75 : 0.45);
    ctx.fillText(s.track.title, box.x, s.sy);
  }
}

export function draw(now) {
  stepTween(now);
  if (!g.frame) return;
  const alpha = ctx.globalAlpha;
  const { x: fx, y: fy, w: fw, h: fh } = g.frame;

  ctx.save();
  ctx.beginPath();
  ctx.rect(fx, fy, fw, fh);
  ctx.clip();
  drawSpecks(alpha);

  const zoomR = clamp(Math.sqrt(g.cam.scale), 1, 3);
  for (const s of g.stars) {
    const [x, y] = toScreen(s.x, s.y);
    const onScreen = x >= fx - 30 && x <= fx + fw + 30 && y >= fy - 30 && y <= fy + fh + 30;
    s.sx = onScreen ? x : NaN;
    s.sy = onScreen ? y : NaN;
  }

  drawRecommendationLinks(alpha);

  // Three passes so recommendations and answers always sit on top of ordinary stars
  const asking = g.answers.size > 0;
  const dimP = appear(now, g.dimChangedAt, 450);
  const fade = g.dimFrom + (g.dimTo - g.dimFrom) * dimP;
  for (const s of g.stars) {
    if (!Number.isFinite(s.sx) || kindOf(s) !== "plain") continue;
    if (s.track.liked) {
      const twinkle = reduceMotion ? 1 : 0.85 + 0.15 * Math.sin(now / 1100 + s.phase * 3);
      const r = 2.3 * zoomR;
      ctx.globalAlpha = alpha * fade * 0.6 * twinkle;
      ctx.drawImage(g.glow, s.sx - r * 4, s.sy - r * 4, r * 8, r * 8);
      ctx.globalAlpha = alpha * fade * twinkle;
      circle(ctx, s.sx, s.sy, r, theme.ink);
    } else {
      ctx.globalAlpha = alpha * fade;
      circle(ctx, s.sx, s.sy, 1.7 * zoomR, theme.starDim);
    }
  }
  for (const s of g.stars) {
    if (!Number.isFinite(s.sx) || kindOf(s) !== "rec") continue;
    const grow = appear(now, g.recsAt, 600, (g.recs.get(s.track.deezer_id).rank - 1) * 110);
    const r = 2.6 * zoomR * grow;
    ctx.globalAlpha = alpha * grow * (asking ? 0.6 : 0.9);
    ctx.drawImage(g.accentGlow, s.sx - r * 4.5, s.sy - r * 4.5, r * 9, r * 9);
    ctx.globalAlpha = alpha * grow * (asking ? 0.6 : 1);
    circle(ctx, s.sx, s.sy, r, theme.accent);
  }
  for (const s of g.stars) {
    if (!Number.isFinite(s.sx) || kindOf(s) !== "answer") continue;
    // Lit up: a large, slowly breathing glow
    const pulse = reduceMotion ? 1 : 0.8 + 0.2 * Math.sin(now / 600 + s.phase);
    const grow = appear(now, g.answersAt, 650, 350 + (g.answers.get(s.track.deezer_id) - 1) * 140);
    const r = 3.4 * zoomR * (0.3 + 0.7 * grow);
    ctx.globalAlpha = alpha * pulse * grow;
    ctx.drawImage(g.glow, s.sx - r * 6, s.sy - r * 6, r * 12, r * 12);
    ctx.globalAlpha = alpha * Math.max(grow, 0.25);
    circle(ctx, s.sx, s.sy, r, theme.ink);
  }

  if (g.cam.scale >= LABEL_FROM_SCALE && !g.tween) drawLabels(alpha);
  drawRanks(alpha);

  // The star under the pointer (or highlighted from a list): accent ring, and the tooltip follows it
  const hovered = g.stars[g.hover];
  if (hovered && Number.isFinite(hovered.sx)) {
    const r = radiusOf(hovered) * zoomR;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(hovered.sx, hovered.sy, r + 6, 0, Math.PI * 2);
    ctx.stroke();
    placeTooltip(hovered);
  }

  ctx.restore();
  ctx.globalAlpha = alpha;
}

function kindOf(star) {
  const id = star.track.deezer_id;
  if (g.answers.has(id)) return "answer";
  if (g.recs.has(id)) return "rec";
  return "plain";
}

function radiusOf(star) {
  const kind = kindOf(star);
  if (kind === "answer") return 3.4;
  if (kind === "rec") return 2.6;
  return star.track.liked ? 2.3 : 1.7;
}

// A thin accent line from each recommended song to the liked song it's closest to
function drawRecommendationLinks(alpha) {
  const hoveredId = g.stars[g.hover]?.track.deezer_id;
  ctx.save();
  ctx.strokeStyle = theme.accent;
  const now = performance.now();
  for (const [id, rec] of g.recs) {
    const a = g.stars[g.index.get(id)];
    const b = g.stars[g.index.get(rec.closestId)];
    if (!a || !b) continue;
    const focused = hoveredId === id || hoveredId === rec.closestId;
    const [ax, ay] = toScreen(a.x, a.y);
    const [bx, by] = toScreen(b.x, b.y);
    const grow = appear(now, g.recsAt, 600, 300 + (rec.rank - 1) * 110);
    ctx.globalAlpha = alpha * grow * (focused ? 0.9 : g.answers.size ? 0.15 : 0.4);
    ctx.lineWidth = focused ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.restore();
}

// Rank numbers matching the "For you" list (accent) and the answer cards (white)
function drawRanks(alpha) {
  ctx.font = font(11, 600);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const zoomR = clamp(Math.sqrt(g.cam.scale), 1, 3);
  const label = (id, text, color, fade) => {
    const s = g.stars[g.index.get(id)];
    if (!s || !Number.isFinite(s.sx)) return;
    ctx.globalAlpha = alpha * fade;
    ctx.fillStyle = color;
    ctx.fillText(text, s.sx + radiusOf(s) * zoomR + 6, s.sy - radiusOf(s) * zoomR - 4);
  };
  const now = performance.now();
  for (const [id, rec] of g.recs) {
    if (!g.answers.has(id)) label(id, String(rec.rank), theme.accent, appear(now, g.recsAt, 400, 400 + rec.rank * 110) * (g.answers.size ? 0.5 : 1));
  }
  for (const [id, rank] of g.answers) label(id, String(rank), theme.ink, appear(now, g.answersAt, 400, 700 + rank * 140));
}

// ---------- recommendations and answers ----------

export function setRecommendations(list) {
  g.recs = new Map(list.map((r, i) => [r.deezer_id, { rank: i + 1, closestId: r.closest_liked.deezer_id }]));
  g.recsAt = performance.now();
}

// Lights up the answer stars and moves the camera so they're all in view above the answer cards
export function setAnswers(ids) {
  const now = performance.now();
  const target = ids.length ? ANSWER_DIM : 1;
  if (target !== g.dimTo) {
    // Fade the rest of the galaxy from wherever it is now toward the new level
    g.dimFrom = g.dimFrom + (g.dimTo - g.dimFrom) * appear(now, g.dimChangedAt, 450);
    g.dimTo = target;
    g.dimChangedAt = now;
  }
  g.answers = new Map(ids.map((id, i) => [id, i + 1]));
  g.answersAt = now;
  const stars = ids.map((id) => g.stars[g.index.get(id)]).filter(Boolean);
  if (!stars.length) return;
  const xs = stars.map((s) => s.x);
  const ys = stars.map((s) => s.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const covered = ui.askBox.getBoundingClientRect().height + 20;   // cards + input over the bottom
  const freeH = Math.max(80, g.area.h - covered);
  const fit = Math.min((g.area.w * 0.7) / Math.max((x1 - x0) * g.base, 1e-6),
                       (freeH * 0.7) / Math.max((y1 - y0) * g.base, 1e-6));
  const scale = clamp(fit, 1, 4);
  // Put the stars' center in the middle of the free space above the cards
  const targetY = g.area.y + freeH / 2;
  tweenTo({ x: (x0 + x1) / 2, y: (y0 + y1) / 2 - (g.cy - targetY) / (g.base * scale), scale }, reduceMotion ? 1 : 900);
}

// Highlight a song from the side panel or a card: same ring and tooltip as hovering its star
export function highlight(id) {
  setHover(id == null ? -1 : g.index.get(id) ?? -1);
}

// ---------- hover ----------

function hitTest(px, py) {
  const reach = HIT_PX * clamp(Math.sqrt(g.cam.scale), 1, 1.6);
  let best = -1;
  let bestD = reach * reach;
  g.stars.forEach((s, i) => {
    if (!Number.isFinite(s.sx)) return;
    const d = (s.sx - px) ** 2 + (s.sy - py) ** 2;
    if (d <= bestD) {
      best = i;
      bestD = d;
    }
  });
  return best;
}

function setHover(index) {
  if (index === g.hover) return;
  g.hover = index;
  const star = g.stars[index];
  if (!star) {
    ui.tooltip.classList.remove("is-visible");
    return;
  }
  const { track } = star;
  ui.tipTitle.textContent = track.title;
  const rec = g.recs.get(track.deezer_id);
  const closest = rec && g.stars[g.index.get(rec.closestId)];
  ui.tipMeta.textContent = `${track.liked ? "♥ " : ""}${track.artist}${track.genre ? " · " + track.genre : ""}` +
    (closest ? ` · for you, like ${closest.track.title}` : "");
  ui.tooltip.classList.add("is-visible");
  g.tipWidth = ui.tooltip.offsetWidth;
}

function placeTooltip(star) {
  const gap = 14;
  const right = star.sx + gap + g.tipWidth < innerWidth - 8;
  const x = right ? star.sx + gap : star.sx - gap - g.tipWidth;
  ui.tooltip.style.transform = `translate(${Math.round(x)}px, ${Math.round(star.sy - gap)}px) translateY(-100%)`;
}

export function clearHover() {
  setHover(-1);
}

// ---------- input ----------

export function pointerDown(e, canvas) {
  canvas.setPointerCapture(e.pointerId);
  g.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  g.tween = null;
  if (g.pointers.size === 1) {
    g.gesture = { type: "pan", moved: 0 };
  } else if (g.pointers.size === 2) {
    const [a, b] = g.pointers.values();
    g.gesture = { type: "pinch", dist: Math.hypot(a.x - b.x, a.y - b.y), mid: [(a.x + b.x) / 2, (a.y + b.y) / 2] };
    clearHover();
  }
}

export function pointerMove(e, canvas) {
  const p = g.pointers.get(e.pointerId);
  if (!p) {
    // Just hovering
    const hit = g.tween ? -1 : hitTest(e.clientX, e.clientY);
    setHover(hit);
    canvas.style.cursor = hit >= 0 ? "pointer" : "grab";
    return;
  }
  const dx = e.clientX - p.x;
  const dy = e.clientY - p.y;
  p.x = e.clientX;
  p.y = e.clientY;

  if (g.gesture?.type === "pan") {
    g.gesture.moved += Math.abs(dx) + Math.abs(dy);
    if (g.gesture.moved >= CLICK_SLOP) {
      pan(dx, dy);
      clearHover();
      canvas.style.cursor = "grabbing";
    }
  } else if (g.gesture?.type === "pinch" && g.pointers.size === 2) {
    const [a, b] = g.pointers.values();
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const mid = [(a.x + b.x) / 2, (a.y + b.y) / 2];
    pan(mid[0] - g.gesture.mid[0], mid[1] - g.gesture.mid[1]);
    if (g.gesture.dist > 0) zoomAt(mid[0], mid[1], dist / g.gesture.dist);
    g.gesture = { type: "pinch", dist, mid };
  }
}

export function pointerUp(e, canvas) {
  const gesture = g.gesture;
  g.pointers.delete(e.pointerId);
  if (g.pointers.size === 0) {
    g.gesture = null;
    canvas.style.cursor = "grab";
    if (gesture?.type === "pan" && gesture.moved < CLICK_SLOP) {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit >= 0) onOpen(g.stars[hit].track);
    }
  } else if (g.pointers.size === 1) {
    // One finger lifted from a pinch: keep panning with the other, but don't treat it as a click
    g.gesture = { type: "pan", moved: CLICK_SLOP };
  }
}

export function pointerLeave() {
  if (!g.pointers.size) clearHover();
}

export function wheel(e) {
  e.preventDefault();
  g.tween = null;
  clearHover();
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;
  // ctrlKey is set for trackpad pinches, which send much smaller deltas
  const speed = e.ctrlKey ? 0.01 : 0.0015;
  zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * unit * speed));
}

// + and - zoom around the center, 0 recenters. Returns true if the key was used.
export function keydown(e) {
  if (e.key === "+" || e.key === "=") zoomAt(g.cx, g.cy, 1.4);
  else if (e.key === "-" || e.key === "_") zoomAt(g.cx, g.cy, 1 / 1.4);
  else if (e.key === "0") recenter();
  else return false;
  return true;
}
