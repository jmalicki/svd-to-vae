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

export type LossPoint = { total: number; recon: number; ortho: number };

function formatAxis(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 100 || a < 0.01) return v.toExponential(1);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function drawLossChart(canvas: HTMLCanvasElement, history: LossPoint[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#F7F7F7";
  ctx.fillRect(0, 0, w, h);

  const left = 44;
  const right = 10;
  const top = 10;
  const bottom = 14;
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

  if (history.length < 2) return;

  const maxY = Math.max(...history.flatMap((p) => [p.total, p.recon, p.ortho]), 1e-9);
  const ticks = [0, 0.5, 1].map((t) => t * maxY);

  ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6b6b6b";

  for (const tick of ticks) {
    const y = top + plotH - (tick / maxY) * plotH;
    ctx.strokeStyle = "rgba(45,45,45,0.08)";
    ctx.beginPath();
    ctx.moveTo(left, y + 0.5);
    ctx.lineTo(left + plotW, y + 0.5);
    ctx.stroke();
    ctx.fillStyle = "#6b6b6b";
    ctx.fillText(formatAxis(tick), left - 6, y);
  }

  // Axis title
  ctx.save();
  ctx.translate(12, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6b6b6b";
  ctx.font = "11px DM Sans, system-ui, sans-serif";
  ctx.fillText("loss", 0, 0);
  ctx.restore();

  const series: { key: keyof LossPoint; color: string }[] = [
    { key: "total", color: "#4C4C4C" },
    { key: "recon", color: "#0072B2" },
    { key: "ortho", color: "#E69F00" },
  ];

  for (const s of series) {
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    history.forEach((p, i) => {
      const x = left + (i / (history.length - 1)) * plotW;
      const y = top + plotH - (p[s.key] / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
