import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import pageHtml from "./pages/svdIntro.html?raw";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import { mountAngleDial } from "./angleDial";
import {
  applyLeft,
  applyMat2,
  applyRight,
  columnOf,
  decomposeAlongNormal,
  demoMatrix2,
  householderAimColumn,
  householderAimToE1,
  householderFromNormal,
  normalFromLineAngle,
  reflectAcrossNormal,
  rightGivensZeroSuperdiag,
} from "./householder2d";
import { prepareHiDpiCanvas } from "./hiDpiCanvas";
import { frameFromMatrix } from "./svdGeometry2d";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
  get,
  fromNested,
  copy,
} from "./matrix";
import { drawHeatmap, drawSigmaBars } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const INK = "#1a1a1a";
const MUTED = "#9a9a9a";
const ACCENT = "#0072B2";
const ACCENT2 = "#E69F00";
const MIRROR = "#D55E00";

const app = document.querySelector<HTMLDivElement>("#app")!;

mountPage(app, pageHtml, {
  nav: chapterNav({
    current: 2,
    prev: { href: "./", label: "← Matrix" },
    next: { href: "./truncate.html", label: "Next →" },
  }),
});

/* ── DOM refs ─────────────────────────────────────────────────────────── */

const el = {
  mirAngVal: app.querySelector<HTMLElement>("#mirAngVal")!,
  mirMirrorDial: app.querySelector<HTMLCanvasElement>("#mirMirrorDial")!,
  mirIn: app.querySelector<HTMLCanvasElement>("#mirIn")!,
  mirOut: app.querySelector<HTMLCanvasElement>("#mirOut")!,
  mirMeterFill: app.querySelector<HTMLElement>("#mirMeterFill")!,
  mirMeterVal: app.querySelector<HTMLElement>("#mirMeterVal")!,
  mirBanner: app.querySelector<HTMLElement>("#mirBanner")!,
  mirHint: app.querySelector<HTMLElement>("#mirHint")!,
  mirPanel: app.querySelector<HTMLElement>("#mirPanel")!,
  buildAngVal: app.querySelector<HTMLElement>("#buildAngVal")!,
  buildProbeAngVal: app.querySelector<HTMLElement>("#buildProbeAngVal")!,
  buildProbeLen: app.querySelector<HTMLInputElement>("#buildProbeLen")!,
  buildProbeLenVal: app.querySelector<HTMLElement>("#buildProbeLenVal")!,
  buildMirrorDial: app.querySelector<HTMLCanvasElement>("#buildMirrorDial")!,
  buildProbeDial: app.querySelector<HTMLCanvasElement>("#buildProbeDial")!,
  build1: app.querySelector<HTMLCanvasElement>("#build1")!,
  build2: app.querySelector<HTMLCanvasElement>("#build2")!,
  build3: app.querySelector<HTMLCanvasElement>("#build3")!,
  buildFormula: app.querySelector<HTMLElement>("#buildFormula")!,
  buildValN: app.querySelector<HTMLElement>("#buildValN")!,
  buildValDot: app.querySelector<HTMLElement>("#buildValDot")!,
  buildValXn: app.querySelector<HTMLElement>("#buildValXn")!,
  buildValHx: app.querySelector<HTMLElement>("#buildValHx")!,

  aimAngVal: app.querySelector<HTMLElement>("#aimAngVal")!,
  aimLen: app.querySelector<HTMLInputElement>("#aimLen")!,
  aimLenVal: app.querySelector<HTMLElement>("#aimLenVal")!,
  aimVecDial: app.querySelector<HTMLCanvasElement>("#aimVecDial")!,
  aimChallenge: app.querySelector<HTMLButtonElement>("#aimChallenge")!,
  aimCanvas: app.querySelector<HTMLCanvasElement>("#aimCanvas")!,
  aimReadout: app.querySelector<HTMLElement>("#aimReadout")!,
  aimSolutionTease: app.querySelector<HTMLElement>("#aimSolutionTease")!,
  aimSolutionBody: app.querySelector<HTMLElement>("#aimSolutionBody")!,
  huntPanel: app.querySelector<HTMLElement>("#huntPanel")!,
  huntAngVal: app.querySelector<HTMLElement>("#huntAngVal")!,
  huntMirrorDial: app.querySelector<HTMLCanvasElement>("#huntMirrorDial")!,
  huntCanvas: app.querySelector<HTMLCanvasElement>("#huntCanvas")!,
  huntHint: app.querySelector<HTMLElement>("#huntHint")!,
  huntBanner: app.querySelector<HTMLElement>("#huntBanner")!,
  huntMeterFill: app.querySelector<HTMLElement>("#huntMeterFill")!,
  huntMeterVal: app.querySelector<HTMLElement>("#huntMeterVal")!,

  hhRegen: app.querySelector<HTMLButtonElement>("#hhRegen")!,
  hhA: app.querySelector<HTMLCanvasElement>("#hhA")!,
  hhHA: app.querySelector<HTMLCanvasElement>("#hhHA")!,
  hhStemIn: app.querySelector<HTMLCanvasElement>("#hhStemIn")!,
  hhStemOut: app.querySelector<HTMLCanvasElement>("#hhStemOut")!,

  eA11: app.querySelector<HTMLInputElement>("#eA11")!,
  eA12: app.querySelector<HTMLInputElement>("#eA12")!,
  eA21: app.querySelector<HTMLInputElement>("#eA21")!,
  eA22: app.querySelector<HTMLInputElement>("#eA22")!,
  eA11Val: app.querySelector<HTMLElement>("#eA11Val")!,
  eA12Val: app.querySelector<HTMLElement>("#eA12Val")!,
  eA21Val: app.querySelector<HTMLElement>("#eA21Val")!,
  eA22Val: app.querySelector<HTMLElement>("#eA22Val")!,
  ellBlend: app.querySelector<HTMLInputElement>("#ellBlend")!,
  ellBlendVal: app.querySelector<HTMLElement>("#ellBlendVal")!,
  ellIn: app.querySelector<HTMLCanvasElement>("#ellIn")!,
  ellOut: app.querySelector<HTMLCanvasElement>("#ellOut")!,
  ellReadout: app.querySelector<HTMLElement>("#ellReadout")!,

  stepLeft: app.querySelector<HTMLButtonElement>("#stepLeft")!,
  stepRight: app.querySelector<HTMLButtonElement>("#stepRight")!,
  stepReset: app.querySelector<HTMLButtonElement>("#stepReset")!,
  stepHelp: app.querySelector<HTMLElement>("#stepHelp")!,
  stepMatrix: app.querySelector<HTMLElement>("#stepMatrix")!,
  stepIn: app.querySelector<HTMLCanvasElement>("#stepIn")!,
  stepOut: app.querySelector<HTMLCanvasElement>("#stepOut")!,

  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  regen: app.querySelector<HTMLButtonElement>("#regen")!,
  A: app.querySelector<HTMLCanvasElement>("#A")!,
  svdU: app.querySelector<HTMLCanvasElement>("#svdU")!,
  svdS: app.querySelector<HTMLCanvasElement>("#svdS")!,
  svdV: app.querySelector<HTMLCanvasElement>("#svdV")!,
  svdRecon: app.querySelector<HTMLCanvasElement>("#svdRecon")!,
  svdErr: app.querySelector<HTMLParagraphElement>("#svdErr")!,
};

/* ── Canvas helpers ────────────────────────────────────────────────────── */

function toCanvas(
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
): [number, number] {
  return [cx + x * scale, cy - y * scale];
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
  color: string,
  label: string,
): void {
  const [x0, y0] = toCanvas(cx, cy, scale, 0, 0);
  const [x1, y1] = toCanvas(cx, cy, scale, x, y);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  const ux = dx / len;
  const uy = dy / len;
  const head = 9;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.25;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * ux + 0.5 * head * uy, y1 - head * uy - 0.5 * head * ux);
  ctx.lineTo(x1 - head * ux - 0.5 * head * uy, y1 - head * uy + 0.5 * head * ux);
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, x1 + 7 * ux, y1 + 7 * uy);
  }
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  sample: (t: number) => [number, number],
  color: string,
  width = 2,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const t = (i / 180) * Math.PI * 2;
    const [x, y] = sample(t);
    const [px, py] = toCanvas(cx, cy, scale, x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawAxes(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  worldR: number,
): void {
  const r = worldR * scale;
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();
  ctx.fillStyle = MUTED;
  ctx.font = "500 11px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("x", cx + r - 12, cy + 14);
  ctx.textAlign = "center";
  ctx.fillText("y", cx + 10, cy - r + 12);
}

function drawMirrorLine(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  lineAngleDeg: number,
  worldR: number,
  opts?: { label?: boolean },
): void {
  const th = (lineAngleDeg * Math.PI) / 180;
  const dx = Math.cos(th);
  const dy = Math.sin(th);
  const r = worldR * 1.15;
  const [x0, y0] = toCanvas(cx, cy, scale, -r * dx, -r * dy);
  const [x1, y1] = toCanvas(cx, cy, scale, r * dx, r * dy);

  // Soft “glass” band so the mirror reads as a surface, not another axis
  const nx = -dy;
  const ny = dx;
  const band = 5;
  ctx.fillStyle = "rgba(213, 94, 0, 0.12)";
  ctx.beginPath();
  ctx.moveTo(x0 + nx * band, y0 - ny * band);
  ctx.lineTo(x1 + nx * band, y1 - ny * band);
  ctx.lineTo(x1 - nx * band, y1 + ny * band);
  ctx.lineTo(x0 - nx * band, y0 + ny * band);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = MIRROR;
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  if (opts?.label !== false) {
    const [lx, ly] = toCanvas(cx, cy, scale, 0.85 * r * dx, 0.85 * r * dy);
    ctx.fillStyle = MIRROR;
    ctx.font = "700 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Offset label off the line along the normal
    ctx.fillText("mirror", lx + nx * 14, ly - ny * 14);
    ctx.textBaseline = "alphabetic";
  }
}

function paintCanvas(
  canvas: HTMLCanvasElement,
  worldR: number,
  drawContent: (ctx: CanvasRenderingContext2D, cx: number, cy: number, scale: number) => void,
): void {
  const { cssW, ctx } = prepareHiDpiCanvas(canvas);
  if (!ctx) return;
  const w = cssW;
  ctx.clearRect(0, 0, w, w);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, w);
  const cx = w / 2;
  const cy = w / 2;
  const scale = (0.4 * w) / worldR;
  drawAxes(ctx, cx, cy, scale, worldR);
  drawContent(ctx, cx, cy, scale);
}

function drawStem(
  canvas: HTMLCanvasElement,
  values: number[],
  color: string,
): void {
  const { cssW, cssH, ctx } = prepareHiDpiCanvas(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 28;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const maxAbsV = Math.max(...values.map((v) => Math.abs(v)), 1e-6);
  const midY = padT + plotH / 2;
  const n = values.length;
  const gap = plotW / Math.max(n, 1);

  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, midY);
  ctx.lineTo(padL + plotW, midY);
  ctx.stroke();

  ctx.font = "600 11px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const x = padL + gap * (i + 0.5);
    const h = (values[i] / maxAbsV) * (plotH * 0.42);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, midY);
    ctx.lineTo(x, midY - h);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, midY - h, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = MUTED;
    ctx.fillText(String(i + 1), x, cssH - 8);
  }
}

function fmt(x: number, d = 2): string {
  return x.toFixed(d);
}

/** Signed number with a figure-space so columns don't jump when the sign flips. */
function fmtSigned(x: number, d = 3): string {
  const body = Math.abs(x).toFixed(d);
  return (x < 0 ? "−" : "\u2007") + body;
}

function fmtPair(x: number, y: number, d = 3): string {
  return `(${fmtSigned(x, d)}, ${fmtSigned(y, d)})`;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/* ── Build reflection operator ─────────────────────────────────────────── */

const NORMAL_COLOR = "#009E73";
/** Normal *component* xₙ — distinct from the unit normal n (green). */
const XN_COLOR = "#56B4E9";
const DECOMP_COLOR = "#7B2D8E";

let mirrorDial: ReturnType<typeof mountAngleDial>;
let probeDial: ReturnType<typeof mountAngleDial>;
let mirDial: ReturnType<typeof mountAngleDial>;
let aimDial: ReturnType<typeof mountAngleDial>;
let huntDial: ReturnType<typeof mountAngleDial>;

function buildProbe(): [number, number] {
  const ang = (probeDial.get() * Math.PI) / 180;
  const len = Number(el.buildProbeLen.value);
  return [len * Math.cos(ang), len * Math.sin(ang)];
}

function paintBuild(): void {
  const ang = mirrorDial.get();
  el.buildAngVal.textContent = `${Math.round(ang)}°`;
  const probeAng = probeDial.get();
  const probeLen = Number(el.buildProbeLen.value);
  el.buildProbeAngVal.textContent = `${Math.round(probeAng)}°`;
  el.buildProbeLenVal.textContent = fmt(probeLen);

  const [nx, ny] = normalFromLineAngle(ang);
  const H = householderFromNormal(nx, ny);
  const [x0, y0] = buildProbe();
  const [hx, hy] = applyMat2(H, x0, y0);
  const { dot, parallel, normal } = decomposeAlongNormal(x0, y0, nx, ny);
  const [xp, yp] = parallel;
  const [xn, yn] = normal;

  el.buildValN.textContent = fmtPair(nx, ny);
  el.buildValDot.textContent = fmtSigned(dot);
  el.buildValXn.textContent = fmtPair(xn, yn);
  el.buildValHx.textContent = fmtPair(hx, hy);

  // Fixed world window so the unit normal n does not appear to change length
  // when the probe slider moves (scale used to track probeLen via extent).
  const extent = 2.5;

  paintCanvas(el.build1, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, nx, ny, NORMAL_COLOR, "n");
  });

  paintCanvas(el.build2, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, xp, yp, DECOMP_COLOR, "x∥");
    // Dashed stub for the normal *component* (a construction piece, not a free vector)
    const [p0x, p0y] = toCanvas(cx, cy, scale, xp, yp);
    const [p1x, p1y] = toCanvas(cx, cy, scale, x0, y0);
    ctx.strokeStyle = XN_COLOR;
    ctx.fillStyle = XN_COLOR;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(p0x, p0y);
    ctx.lineTo(p1x, p1y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(p1x, p1y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.fillText("xₙ", p1x + 6, p1y - 6);
    drawArrow(ctx, cx, cy, scale, x0, y0, ACCENT, "x");
  });

  paintCanvas(el.build3, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, x0, y0, ACCENT, "x");
    drawArrow(ctx, cx, cy, scale, hx, hy, ACCENT2, "Hx");
    ctx.strokeStyle = MUTED;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const [a0, a1] = toCanvas(cx, cy, scale, x0, y0);
    const [b0, b1] = toCanvas(cx, cy, scale, hx, hy);
    ctx.moveTo(a0, a1);
    ctx.lineTo(b0, b1);
    ctx.stroke();
    ctx.setLineDash([]);
    const [fx, fy] = toCanvas(cx, cy, scale, xp, yp);
    ctx.fillStyle = MUTED;
    ctx.beginPath();
    ctx.arc(fx, fy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

/* ── Reflect across a line (match target) ──────────────────────────────── */

/** Orange arrow to flip; dashed blue = where it should land. */
const MIR_SRC: [number, number] = [1, 0];
const MIR_TGT: [number, number] = [0, 1];

function drawDashedArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
  color: string,
  label: string,
): void {
  const [x0, y0] = toCanvas(cx, cy, scale, 0, 0);
  const [x1, y1] = toCanvas(cx, cy, scale, x, y);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  const ux = dx / len;
  const uy = dy / len;
  const head = 9;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.25;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * ux + 0.5 * head * uy, y1 - head * uy - 0.5 * head * ux);
  ctx.lineTo(x1 - head * ux - 0.5 * head * uy, y1 - head * uy + 0.5 * head * ux);
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, x1 + 7 * ux, y1 + 7 * uy);
  }
}

function paintMirror(): void {
  const ang = mirDial.get();
  el.mirAngVal.textContent = `${Math.round(ang)}°`;
  const [nx, ny] = normalFromLineAngle(ang);
  const H = householderFromNormal(nx, ny);
  const [rx, ry] = applyMat2(H, MIR_SRC[0], MIR_SRC[1]);
  const dist = Math.hypot(rx - MIR_TGT[0], ry - MIR_TGT[1]);
  const won = dist < 0.08;

  const meter = Math.min(1, dist / 1.5);
  el.mirMeterFill.style.transform = `scaleX(${meter})`;
  el.mirMeterFill.classList.toggle("near", meter < 0.25 && !won);
  el.mirMeterFill.classList.toggle("won", won);
  el.mirMeterVal.textContent = fmt(dist, 3);
  el.mirBanner.hidden = !won;
  el.mirPanel.classList.toggle("hunt-won", won);
  el.mirHint.textContent = won
    ? "Hx = y — you solved the inverse: n for this prescribed image."
    : "Find n so Hx hits y (inverse of “given n, compute Hx”).";

  paintCanvas(el.mirIn, 1.4, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, 1.4);
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, MIR_SRC[0], MIR_SRC[1], ACCENT, "x");
    drawDashedArrow(ctx, cx, cy, scale, MIR_TGT[0], MIR_TGT[1], ACCENT, "y");
  });

  paintCanvas(el.mirOut, 1.4, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, 1.4);
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (t) => reflectAcrossNormal(Math.cos(t), Math.sin(t), nx, ny),
      INK,
    );
    drawDashedArrow(ctx, cx, cy, scale, MIR_TGT[0], MIR_TGT[1], ACCENT, "y");
    // Blue Hx (not orange — that color is reserved for the mirror band)
    drawArrow(ctx, cx, cy, scale, rx, ry, won ? "#009E73" : ACCENT, "Hx");
    if (won) {
      // Soft green halo under the matched tip
      const [tx, ty] = toCanvas(cx, cy, scale, MIR_TGT[0], MIR_TGT[1]);
      ctx.fillStyle = "rgba(0, 158, 115, 0.18)";
      ctx.beginPath();
      ctx.arc(tx, ty, 10, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/* ── Aim onto axis (challenge) ─────────────────────────────────────────── */

let huntWon = false;

function aimVector(): [number, number] {
  const th = (aimDial.get() * Math.PI) / 180;
  const len = Number(el.aimLen.value);
  return [len * Math.cos(th), len * Math.sin(th)];
}

/** Offset hunt mirror from the true Householder so the challenge is not already solved. */
function resetHuntAwayFromAnswer(answerDeg: number): void {
  let offset = 35 + Math.floor(Math.random() * 40);
  if (Math.random() < 0.5) offset = -offset;
  let h = answerDeg + offset;
  while (h > 90) h -= 180;
  while (h < -90) h += 180;
  huntDial.set(h, true);
  huntWon = false;
}

function newAimChallenge(): void {
  aimDial.set(Math.round(-150 + Math.random() * 300), true);
  el.aimLen.value = (0.6 + Math.random() * 1.4).toFixed(2);
  const [ax, ay] = aimVector();
  const aimed = householderAimToE1(ax, ay);
  resetHuntAwayFromAnswer(aimed.mirrorAngleDeg);
  paintAim();
}

function paintAim(): void {
  const ang = aimDial.get();
  const len = Number(el.aimLen.value);
  el.aimAngVal.textContent = `${Math.round(ang)}°`;
  el.aimLenVal.textContent = fmt(len);

  const [ax, ay] = aimVector();
  const aimed = householderAimToE1(ax, ay);
  const worldR = Math.max(2.0, aimed.normA * 1.25);

  const hunt = huntDial.get();
  el.huntAngVal.textContent = `${Math.round(hunt)}°`;
  const [hnx, hny] = normalFromLineAngle(hunt);
  const [rx, ry] = reflectAcrossNormal(ax, ay, hnx, hny);
  const height = Math.abs(ry);
  const thresh = 0.045 * Math.max(aimed.normA, 1);
  const onAxis = height < thresh;
  if (onAxis && !huntWon) huntWon = true;

  // Proximity meter: full when far, empty when on axis
  const meter = Math.min(1, height / Math.max(aimed.normA, 0.5));
  el.huntMeterFill.style.transform = `scaleX(${meter})`;
  el.huntMeterFill.classList.toggle("near", meter < 0.25 && !onAxis);
  el.huntMeterFill.classList.toggle("won", onAxis);
  el.huntMeterVal.textContent = fmt(height, 3);
  el.huntPanel.classList.toggle("hunt-won", onAxis);
  el.huntBanner.hidden = !onAxis;

  paintCanvas(el.huntCanvas, worldR, (ctx, cx, cy, scale) => {
    if (onAxis) {
      ctx.strokeStyle = "#009E73";
      ctx.lineWidth = 4;
      ctx.beginPath();
      const [x0, y0] = toCanvas(cx, cy, scale, -worldR, 0);
      const [x1, y1] = toCanvas(cx, cy, scale, worldR, 0);
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    drawMirrorLine(ctx, cx, cy, scale, hunt, worldR);
    drawArrow(ctx, cx, cy, scale, ax, ay, ACCENT2, "a");
    drawArrow(ctx, cx, cy, scale, rx, ry, onAxis ? "#009E73" : ACCENT, "Ha");
  });

  if (onAxis) {
    el.huntHint.textContent =
      `Ha ≈ (${fmt(rx, 3)}, ${fmt(ry, 3)}). This n matches n ∥ (a − t) below.`;
    el.aimSolutionTease.hidden = true;
    el.aimSolutionBody.hidden = false;
  } else {
    const warmth =
      meter < 0.15 ? "Almost: |(Ha)₂| is nearly 0." :
      meter < 0.35 ? "n is getting closer." :
      "Vary n; watch |(Ha)₂|.";
    el.huntHint.textContent = warmth;
    if (!huntWon) {
      el.aimSolutionTease.hidden = false;
      el.aimSolutionBody.hidden = true;
    }
  }

  paintCanvas(el.aimCanvas, worldR, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, aimed.mirrorAngleDeg, worldR);
    const [sx, sy] = toCanvas(cx, cy, scale, ax, ay);
    const [tx, ty] = toCanvas(cx, cy, scale, aimed.targetX, aimed.targetY);
    ctx.strokeStyle = MUTED;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    drawArrow(ctx, cx, cy, scale, ax, ay, ACCENT2, "a");
    drawArrow(ctx, cx, cy, scale, aimed.targetX, aimed.targetY, ACCENT, "target");
    const [hx, hy] = applyMat2(aimed.H, ax, ay);
    drawArrow(ctx, cx, cy, scale, hx, hy, INK, "Ha");
  });

  el.aimReadout.textContent =
    `t = (${fmt(aimed.targetX, 3)}, 0). n ≈ (${fmt(aimed.nx, 3)}, ${fmt(aimed.ny, 3)}). ` +
    `Ha ≈ (${fmt(aimed.targetX, 3)}, 0).`;
}

/* ── 5×5 Householder ───────────────────────────────────────────────────── */

let hhMat: Matrix;
let hhHA: Matrix;

function regenHH(): void {
  hhMat = randomNormal(5, 5, 1);
  const H = householderAimColumn(columnOf(hhMat, 0));
  hhHA = applyLeft(H, hhMat);
  paintHH();
}

function paintHH(): void {
  const scale = Math.max(maxAbs(hhMat), maxAbs(hhHA), 1e-6);
  drawHeatmap(el.hhA, hhMat, scale);
  drawHeatmap(el.hhHA, hhHA, scale);
  drawStem(el.hhStemIn, columnOf(hhMat, 0), ACCENT2);
  drawStem(el.hhStemOut, columnOf(hhHA, 0), ACCENT);
}

/* ── Ellipse under left HH ─────────────────────────────────────────────── */

function readEllipseA(): Matrix {
  return fromNested([
    [Number(el.eA11.value), Number(el.eA12.value)],
    [Number(el.eA21.value), Number(el.eA22.value)],
  ]);
}

function syncEllipseLabels(): void {
  el.eA11Val.textContent = fmt(Number(el.eA11.value));
  el.eA12Val.textContent = fmt(Number(el.eA12.value));
  el.eA21Val.textContent = fmt(Number(el.eA21.value));
  el.eA22Val.textContent = fmt(Number(el.eA22.value));
  el.ellBlendVal.textContent = fmt(Number(el.ellBlend.value));
}

function paintEllipse(): void {
  syncEllipseLabels();
  const A = readEllipseA();
  const col = columnOf(A, 0);
  const { H } = householderAimToE1(col[0], col[1]);
  const HA = applyLeft(H, A);
  const t = Number(el.ellBlend.value);
  const fA = frameFromMatrix(A);
  const fHA = frameFromMatrix(HA);
  const [s1, s2] = fA.sigma;
  const outR = Math.max(fA.outScale, fHA.outScale);

  paintCanvas(el.ellIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (u) => [Math.cos(u), Math.sin(u)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1,0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0,1)");
  });

  paintCanvas(el.ellOut, outR, (ctx, cx, cy, scale) => {
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (u) => {
        const c = Math.cos(u);
        const s = Math.sin(u);
        const aPt = applyMat2(A, c, s);
        const hPt = applyMat2(HA, c, s);
        return [
          aPt[0] * (1 - t) + hPt[0] * t,
          aPt[1] * (1 - t) + hPt[1] * t,
        ];
      },
      INK,
    );
    const a10 = applyMat2(A, 1, 0);
    const h10 = applyMat2(HA, 1, 0);
    const ix = a10[0] * (1 - t) + h10[0] * t;
    const iy = a10[1] * (1 - t) + h10[1] * t;
    drawArrow(ctx, cx, cy, scale, ix, iy, ACCENT2, "A(1,0)→");
    if (t > 0.85) {
      drawArrow(ctx, cx, cy, scale, h10[0], h10[1], ACCENT, "HA(1,0)");
    }
  });

  el.ellReadout.textContent =
    `Stretch amounts σ ≈ ${fmt(s1, 3)}, ${fmt(s2, 3)} — unchanged by left H ` +
    `(HA has σ ≈ ${fmt(fHA.sigma[0], 3)}, ${fmt(fHA.sigma[1], 3)}). ` +
    `At blend 1, image of (1,0) lies on the x-axis.`;
}

/* ── Left / right steps ────────────────────────────────────────────────── */

let stepM: Matrix;
let stepPhase: "start" | "left" | "done" = "start";

function fmtMat2(M: Matrix): string {
  return (
    `$$\\begin{pmatrix}` +
    `${fmt(get(M, 0, 0), 3)} & ${fmt(get(M, 0, 1), 3)} \\\\ ` +
    `${fmt(get(M, 1, 0), 3)} & ${fmt(get(M, 1, 1), 3)}` +
    `\\end{pmatrix}$$`
  );
}

function resetSteps(): void {
  stepM = copy(demoMatrix2());
  stepPhase = "start";
  el.stepHelp.textContent = "Start from the demo 2×2. Apply left H, then right G.";
  el.stepLeft.disabled = false;
  el.stepRight.disabled = true;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

function paintSteps(): void {
  const f = frameFromMatrix(stepM);
  const outR = f.outScale;

  paintCanvas(el.stepIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (u) => [Math.cos(u), Math.sin(u)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1,0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0,1)");
  });

  paintCanvas(el.stepOut, outR, (ctx, cx, cy, scale) => {
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (u) => applyMat2(stepM, Math.cos(u), Math.sin(u)),
      INK,
    );
    const [s1, s2] = f.sigma;
    drawArrow(
      ctx,
      cx,
      cy,
      scale,
      get(f.U, 0, 0) * s1,
      get(f.U, 1, 0) * s1,
      ACCENT2,
      "σ₁",
    );
    if (s2 > 1e-6) {
      drawArrow(
        ctx,
        cx,
        cy,
        scale,
        get(f.U, 0, 1) * s2,
        get(f.U, 1, 1) * s2,
        ACCENT,
        "σ₂",
      );
    }
  });

  const phaseLabel =
    stepPhase === "start"
      ? "Current $A$"
      : stepPhase === "left"
        ? "After left $H$"
        : "After left $H$ and right $G$";
  el.stepMatrix.innerHTML = `${phaseLabel}: ${fmtMat2(stepM)}`;
}

function doStepLeft(): void {
  if (stepPhase !== "start") return;
  const { H } = householderAimToE1(get(stepM, 0, 0), get(stepM, 1, 0));
  stepM = applyLeft(H, stepM);
  stepPhase = "left";
  el.stepHelp.textContent = "Column 1 on the axis. Next: right Givens to zero the (1,2) entry.";
  el.stepLeft.disabled = true;
  el.stepRight.disabled = false;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

function doStepRight(): void {
  if (stepPhase !== "left") return;
  const G = rightGivensZeroSuperdiag(stepM);
  stepM = applyRight(stepM, G);
  stepPhase = "done";
  el.stepHelp.textContent =
    "Nearly diagonal — absolute diagonal entries match the ellipse stretches.";
  el.stepRight.disabled = true;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

/* ── Any-size SVD heatmaps ─────────────────────────────────────────────── */

const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

let A: Matrix;
let svd: SvdResult;
let svdReconMat: Matrix;
let sharedScale = 1;
let sigmaScale = 1;

function sizeFromSlider(): number {
  const i = clamp(Math.round(Number(el.size.value) || 0), 0, SIZE_STOPS.length - 1);
  el.size.value = String(i);
  return SIZE_STOPS[i];
}

function syncSliderLabels(): void {
  el.sizeVal.textContent = String(sizeFromSlider());
}

function recompute(newA: boolean): void {
  const n = sizeFromSlider();
  syncSliderLabels();
  if (newA || !A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, n);
  svdReconMat = reconstruct(svd.U, svd.sigma, svd.V);
  sharedScale = Math.max(maxAbs(A), maxAbs(svdReconMat), 1e-6);
  sigmaScale = Math.max(...svd.sigma, 1e-6);
  paintSvd();
}

function paintSvd(): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdReconMat, sharedScale);
  const err = frobeniusSq(sub(A, svdReconMat));
  el.svdErr.textContent = `‖A − product‖_F² = ${err.toExponential(3)}`;
}

/* ── Master paint + listeners ──────────────────────────────────────────── */

mirrorDial = mountAngleDial(el.buildMirrorDial, {
  kind: "line",
  periodDeg: 180,
  valueDeg: 35,
  color: MIRROR,
  onChange: () => paintBuild(),
});
probeDial = mountAngleDial(el.buildProbeDial, {
  kind: "ray",
  periodDeg: 360,
  valueDeg: 27,
  color: ACCENT,
  onChange: () => paintBuild(),
});
mirDial = mountAngleDial(el.mirMirrorDial, {
  kind: "line",
  periodDeg: 180,
  valueDeg: 10,
  color: MIRROR,
  onChange: () => paintMirror(),
});
aimDial = mountAngleDial(el.aimVecDial, {
  kind: "ray",
  periodDeg: 360,
  valueDeg: 55,
  color: ACCENT2,
  onChange: () => {
    huntWon = false;
    paintAim();
  },
});
huntDial = mountAngleDial(el.huntMirrorDial, {
  kind: "line",
  periodDeg: 180,
  valueDeg: 20,
  color: MIRROR,
  onChange: () => paintAim(),
});

function paintAll(): void {
  paintBuild();
  paintMirror();
  paintAim();
  paintHH();
  paintEllipse();
  paintSteps();
  paintSvd();
}

el.buildProbeLen.addEventListener("input", paintBuild);

el.aimChallenge.addEventListener("click", newAimChallenge);
el.aimLen.addEventListener("input", () => {
  huntWon = false;
  paintAim();
});

el.hhRegen.addEventListener("click", regenHH);

for (const input of [el.eA11, el.eA12, el.eA21, el.eA22, el.ellBlend]) {
  input.addEventListener("input", paintEllipse);
}

el.stepLeft.addEventListener("click", doStepLeft);
el.stepRight.addEventListener("click", doStepRight);
el.stepReset.addEventListener("click", resetSteps);

el.regen.addEventListener("click", () => recompute(true));
el.size.addEventListener("input", () => recompute(true));

regenHH();
resetSteps();
syncSliderLabels();
recompute(true);
{
  const [ax, ay] = aimVector();
  resetHuntAwayFromAnswer(householderAimToE1(ax, ay).mirrorAngleDeg);
}
paintAll();

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});
