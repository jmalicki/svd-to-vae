import { type Matrix, maxAbs, get } from "./matrix";

// ggplot-like diverging scale: steel blue ↔ cream ↔ amber (colorblind-friendlier than red–green)
const POS = [230, 159, 0]; // #E69F00
const NEG = [0, 114, 178]; // #0072B2
const ZERO = [247, 247, 247]; // #F7F7F7

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function colorFor(v: number, scale: number): string {
  const t = Math.max(-1, Math.min(1, v / scale));
  if (t >= 0) {
    const u = t;
    return `rgb(${lerp(ZERO[0], POS[0], u) | 0},${lerp(ZERO[1], POS[1], u) | 0},${lerp(ZERO[2], POS[2], u) | 0})`;
  }
  const u = -t;
  return `rgb(${lerp(ZERO[0], NEG[0], u) | 0},${lerp(ZERO[1], NEG[1], u) | 0},${lerp(ZERO[2], NEG[2], u) | 0})`;
}

export function drawHeatmap(
  canvas: HTMLCanvasElement,
  m: Matrix,
  sharedScale?: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const scale = sharedScale ?? maxAbs(m);
  const cell = Math.max(4, Math.floor(Math.min(canvas.width / m.cols, canvas.height / m.rows)));
  const w = cell * m.cols;
  const h = cell * m.rows;
  canvas.width = w;
  canvas.height = h;
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      ctx.fillStyle = colorFor(get(m, i, j), scale);
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  }
  ctx.strokeStyle = "rgba(45,45,45,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= m.rows; i++) {
    ctx.beginPath();
    ctx.moveTo(0, i * cell + 0.5);
    ctx.lineTo(w, i * cell + 0.5);
    ctx.stroke();
  }
  for (let j = 0; j <= m.cols; j++) {
    ctx.beginPath();
    ctx.moveTo(j * cell + 0.5, 0);
    ctx.lineTo(j * cell + 0.5, h);
    ctx.stroke();
  }
}

export function drawSigmaBars(canvas: HTMLCanvasElement, sigma: number[], maxS?: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const n = Math.max(1, sigma.length);
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const peak = maxS ?? Math.max(...sigma.map(Math.abs), 1e-9);
  const gap = 4;
  const barW = (w - gap * (n + 1)) / n;
  for (let i = 0; i < n; i++) {
    const bh = (Math.abs(sigma[i]) / peak) * (h - 8);
    const x = gap + i * (barW + gap);
    const y = h - 4 - bh;
    ctx.fillStyle = `rgb(${POS[0]},${POS[1]},${POS[2]})`;
    ctx.fillRect(x, y, barW, bh);
  }
}

export type LossPoint = { step: number; recon: number; vsSvd: number };

const LOG_EPS = 1e-18;

function formatAxis(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 100 || a < 0.01) return v.toExponential(0);
  if (a >= 10) return v.toFixed(0);
  return v.toFixed(2);
}

function logClamp(v: number): number {
  return Math.log10(Math.max(v, LOG_EPS));
}

/** Decade ticks from lo..hi (inclusive), plus endpoints if needed. */
function logTicks(logMin: number, logMax: number): number[] {
  const lo = Math.ceil(logMin - 1e-12);
  const hi = Math.floor(logMax + 1e-12);
  const ticks: number[] = [];
  if (hi - lo + 1 <= 6) {
    for (let e = lo; e <= hi; e++) ticks.push(e);
  } else {
    const step = Math.ceil((hi - lo) / 4);
    for (let e = lo; e <= hi; e += step) ticks.push(e);
    if (ticks[ticks.length - 1] !== hi) ticks.push(hi);
  }
  if (ticks.length === 0) {
    ticks.push(Math.floor(logMin), Math.ceil(logMax));
  }
  return ticks;
}

function strokeSeries(
  ctx: CanvasRenderingContext2D,
  history: LossPoint[],
  key: "recon" | "vsSvd",
  maxStep: number,
  left: number,
  plotW: number,
  yAt: (v: number) => number,
  color: string,
): void {
  if (history.length < 2) {
    const p = history[0];
    const x = left + (p.step / maxStep) * plotW;
    const y = yAt(p[key]);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.75;
  history.forEach((p, i) => {
    const x = left + (p.step / maxStep) * plotW;
    const y = yAt(p[key]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export function drawLossChart(
  canvas: HTMLCanvasElement,
  history: LossPoint[],
  svdRecon: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#F7F7F7";
  ctx.fillRect(0, 0, w, h);

  const left = 52;
  const right = 10;
  const top = 10;
  const bottom = 22;
  const plotW = w - left - right;
  const plotH = h - top - bottom;

  // Axis spine
  ctx.strokeStyle = "rgba(45,45,45,0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left + 0.5, top);
  ctx.lineTo(left + 0.5, top + plotH);
  ctx.lineTo(left + plotW, top + plotH + 0.5);
  ctx.stroke();

  if (history.length < 1) return;

  const maxStep = Math.max(history[history.length - 1]?.step ?? 0, 1);
  const vals = [
    ...history.map((p) => p.recon),
    ...history.map((p) => p.vsSvd),
    svdRecon,
  ].map((v) => Math.max(v, LOG_EPS));
  let logMax = Math.log10(Math.max(...vals));
  let logMin = Math.log10(Math.min(...vals));
  if (logMax - logMin < 0.5) {
    const mid = 0.5 * (logMax + logMin);
    logMin = mid - 0.5;
    logMax = mid + 0.5;
  } else {
    const pad = 0.08 * (logMax - logMin);
    logMin -= pad;
    logMax += pad;
  }

  const yAt = (v: number) =>
    top + plotH - ((logClamp(v) - logMin) / (logMax - logMin)) * plotH;

  const ticks = logTicks(logMin, logMax);

  ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6b6b6b";

  for (const e of ticks) {
    const y = top + plotH - ((e - logMin) / (logMax - logMin)) * plotH;
    if (y < top - 2 || y > top + plotH + 2) continue;
    ctx.strokeStyle = "rgba(45,45,45,0.08)";
    ctx.beginPath();
    ctx.moveTo(left, y + 0.5);
    ctx.lineTo(left + plotW, y + 0.5);
    ctx.stroke();
    ctx.fillStyle = "#6b6b6b";
    ctx.fillText(formatAxis(10 ** e), left - 6, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = "#6b6b6b";
  ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  ctx.fillText("0", left, top + plotH + 6);
  ctx.fillText(String(maxStep), left + plotW, top + plotH + 6);
  ctx.font = "11px DM Sans, system-ui, sans-serif";
  ctx.fillText("step", left + plotW / 2, top + plotH + 6);

  ctx.save();
  ctx.translate(14, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6b6b6b";
  ctx.font = "11px DM Sans, system-ui, sans-serif";
  ctx.fillText("‖·‖² (log)", 0, 0);
  ctx.restore();

  // Classical SVD: ‖A − Â_svd‖² (constant)
  const ySvd = yAt(svdRecon);
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#4C4C4C";
  ctx.lineWidth = 1.75;
  ctx.beginPath();
  ctx.moveTo(left, ySvd + 0.5);
  ctx.lineTo(left + plotW, ySvd + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  // GD: ‖A − Â_gd‖²
  strokeSeries(ctx, history, "recon", maxStep, left, plotW, yAt, "#0072B2");
  // Gap between reconstructions: ‖Â_svd − Â_gd‖²
  strokeSeries(ctx, history, "vsSvd", maxStep, left, plotW, yAt, "#E69F00");
}
