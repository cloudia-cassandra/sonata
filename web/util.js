// Shared by the galaxy and the song sky.

// Colors and the font come from the CSS variables in style.css (the single source of the
// design), read once at startup, so canvas drawing always matches the HTML around it.
export const theme = {};

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  const full = value.length === 3 ? [...value].map((c) => c + c).join("") : value;
  const n = parseInt(full, 16);
  return `${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}`;
}

export function loadTheme() {
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  Object.assign(theme, {
    ink: token("--ink"),
    muted: token("--muted"),
    faint: token("--faint"),
    starDim: token("--star-dim"),
    accent: token("--accent"),
    font: token("--font"),
  });
  theme.inkRgb = hexToRgb(theme.ink);
  theme.accentRgb = hexToRgb(theme.accent);
}

export const font = (size, weight = 400) => `${weight} ${size}px ${theme.font}`;

export const ink = (alpha) => `rgba(${theme.inkRgb}, ${alpha})`;

export const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export const easeInOut = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

export const easeOut = (p) => 1 - Math.pow(1 - p, 3);

// 0 -> 1 over `duration` ms after `start` (+ delay); always 1 with reduced motion
export function appear(now, start, duration, delay = 0) {
  if (reduceMotion) return 1;
  return easeOut(clamp((now - start - delay) / duration, 0, 1));
}

// Small seeded random generator, so decorative specks are the same on every visit
export function seededRandom(seed) {
  let a = seed % 4294967296;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function circle(ctx, x, y, r, fill) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
