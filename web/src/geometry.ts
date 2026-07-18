import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import pageHtml from "./pages/geometry.html?raw";
import {
  applyMat2,
  frameFromFactors,
  frameFromMatrix,
  rotationMatrixDeg,
  stretchMatrix,
  type AngleFactors,
  type EllipseFrame,
} from "./svdGeometry2d";
import { fromNested, get, type Matrix } from "./matrix";
import { prepareHiDpiCanvas } from "./hiDpiCanvas";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

mountPage(app, pageHtml, {
  nav: chapterNav({ current: 1, next: { href: "./reflections.html", label: "Next →" } }),
});

type Factors = AngleFactors;

const A_PRESETS: Record<string, [number, number, number, number]> = {
  default: [1.2, 0.9, 0.4, 1.5],
  example: [1.5, 1.0, 0.2, 1.2],
  diag: [2.0, 0, 0, 0.6],
  flat: [1.8, 0.9, 0.85, 0.45],
};

const SYNTH_PRESETS: Record<string, Factors> = {
  stretch: { s1: 2.4, s2: 0.7, thVDeg: 0, thUDeg: 0 },
  circle: { s1: 1.4, s2: 1.4, thVDeg: 0, thUDeg: 35 },
  flat: { s1: 2.2, s2: 0, thVDeg: 20, thUDeg: 55 },
  tilt: { s1: 2.0, s2: 0.9, thVDeg: 35, thUDeg: -20 },
};

const el = {
  a11: app.querySelector<HTMLInputElement>("#a11")!,
  a12: app.querySelector<HTMLInputElement>("#a12")!,
  a21: app.querySelector<HTMLInputElement>("#a21")!,
  a22: app.querySelector<HTMLInputElement>("#a22")!,
  a11Val: app.querySelector<HTMLElement>("#a11Val")!,
  a12Val: app.querySelector<HTMLElement>("#a12Val")!,
  a21Val: app.querySelector<HTMLElement>("#a21Val")!,
  a22Val: app.querySelector<HTMLElement>("#a22Val")!,
  analysisA: app.querySelector<HTMLElement>("#analysisA")!,
  stretchReadout: app.querySelector<HTMLElement>("#stretchReadout")!,
  rotDeg: app.querySelector<HTMLInputElement>("#rotDeg")!,
  rotDegVal: app.querySelector<HTMLElement>("#rotDegVal")!,
  rotFormula: app.querySelector<HTMLElement>("#rotFormula")!,
  sx: app.querySelector<HTMLInputElement>("#sx")!,
  sy: app.querySelector<HTMLInputElement>("#sy")!,
  sxVal: app.querySelector<HTMLElement>("#sxVal")!,
  syVal: app.querySelector<HTMLElement>("#syVal")!,
  stFormula: app.querySelector<HTMLElement>("#stFormula")!,
  stHint: app.querySelector<HTMLElement>("#stHint")!,
  rotIn: app.querySelector<HTMLCanvasElement>("#rotIn")!,
  rotOut: app.querySelector<HTMLCanvasElement>("#rotOut")!,
  stIn: app.querySelector<HTMLCanvasElement>("#stIn")!,
  stOut: app.querySelector<HTMLCanvasElement>("#stOut")!,
  anIn: app.querySelector<HTMLCanvasElement>("#anIn")!,
  anOut: app.querySelector<HTMLCanvasElement>("#anOut")!,
  mv0: app.querySelector<HTMLCanvasElement>("#mv0")!,
  mv1: app.querySelector<HTMLCanvasElement>("#mv1")!,
  mv2: app.querySelector<HTMLCanvasElement>("#mv2")!,
  mv3: app.querySelector<HTMLCanvasElement>("#mv3")!,
  s1: app.querySelector<HTMLInputElement>("#s1")!,
  s2: app.querySelector<HTMLInputElement>("#s2")!,
  thV: app.querySelector<HTMLInputElement>("#thV")!,
  thU: app.querySelector<HTMLInputElement>("#thU")!,
  s1Val: app.querySelector<HTMLElement>("#s1Val")!,
  s2Val: app.querySelector<HTMLElement>("#s2Val")!,
  thVVal: app.querySelector<HTMLElement>("#thVVal")!,
  thUVal: app.querySelector<HTMLElement>("#thUVal")!,
  synthA: app.querySelector<HTMLElement>("#synthA")!,
  synthReadout: app.querySelector<HTMLElement>("#synthReadout")!,
  syIn: app.querySelector<HTMLCanvasElement>("#syIn")!,
  syOut: app.querySelector<HTMLCanvasElement>("#syOut")!,
};

const ACCENT = "#0072b2";
const ACCENT2 = "#e69f00";
const INK = "#2d2d2d";
const MUTED = "rgba(45,45,45,0.35)";

function fmt(v: number): string {
  return v.toFixed(2);
}

function readA(): Matrix {
  return fromNested([
    [Number(el.a11.value), Number(el.a12.value)],
    [Number(el.a21.value), Number(el.a22.value)],
  ]);
}

function setA(a11: number, a12: number, a21: number, a22: number): void {
  el.a11.value = String(a11);
  el.a12.value = String(a12);
  el.a21.value = String(a21);
  el.a22.value = String(a22);
}

function readSynth(): Factors {
  return {
    s1: Number(el.s1.value),
    s2: Number(el.s2.value),
    thVDeg: Number(el.thV.value),
    thUDeg: Number(el.thU.value),
  };
}

function setSynth(f: Factors): void {
  el.s1.value = String(f.s1);
  el.s2.value = String(f.s2);
  el.thV.value = String(f.thVDeg);
  el.thU.value = String(f.thUDeg);
}

let analysisFrame: EllipseFrame | null = null;
let synthFrame: EllipseFrame | null = null;

function rebuildAnalysis(): EllipseFrame {
  analysisFrame = frameFromMatrix(readA());
  return analysisFrame;
}

function rebuildSynth(): EllipseFrame {
  synthFrame = frameFromFactors(readSynth());
  return synthFrame;
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
}

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
  ctx.font = "600 12px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, x1 + 7 * ux, y1 + 7 * uy);
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  sample: (t: number) => [number, number],
  color: string,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
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

function paintPair(
  inC: HTMLCanvasElement,
  outC: HTMLCanvasElement,
  f: EllipseFrame,
  probe: number,
  labels: { v1: string; v2: string; u1: string; u2: string },
): void {
  const { A, U, V, sigma, outScale } = f;
  paintCanvas(inC, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, get(V, 0, 0), get(V, 1, 0), ACCENT2, labels.v1);
    drawArrow(ctx, cx, cy, scale, get(V, 0, 1), get(V, 1, 1), ACCENT, labels.v2);
    const [qx, qy] = toCanvas(cx, cy, scale, Math.cos(probe), Math.sin(probe));
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(qx, qy, 4.5, 0, Math.PI * 2);
    ctx.fill();
  });
  paintCanvas(outC, outScale, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(A, Math.cos(t), Math.sin(t)), INK);
    const [s1, s2] = sigma;
    drawArrow(ctx, cx, cy, scale, get(U, 0, 0) * s1, get(U, 1, 0) * s1, ACCENT2, labels.u1);
    if (s2 > 1e-6) {
      drawArrow(ctx, cx, cy, scale, get(U, 0, 1) * s2, get(U, 1, 1) * s2, ACCENT, labels.u2);
    }
    const [ax, ay] = applyMat2(A, Math.cos(probe), Math.sin(probe));
    const [qx, qy] = toCanvas(cx, cy, scale, ax, ay);
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(qx, qy, 4.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function paintPieces(): void {
  const deg = Number(el.rotDeg.value);
  const sx = Number(el.sx.value);
  const sy = Number(el.sy.value);
  const ROT_A = rotationMatrixDeg(deg);
  const STRETCH_A = stretchMatrix(sx, sy);

  el.rotDegVal.textContent = `${Math.round(deg)}°`;
  el.sxVal.textContent = fmt(sx);
  el.syVal.textContent = fmt(sy);
  el.rotFormula.textContent =
    `R = [[${fmt(get(ROT_A, 0, 0))}, ${fmt(get(ROT_A, 0, 1))}],  [${fmt(get(ROT_A, 1, 0))}, ${fmt(get(ROT_A, 1, 1))}]]  (${Math.round(deg)}°)`;
  el.stFormula.textContent = `S = [[${fmt(sx)}, 0],  [0, ${fmt(sy)}]]`;
  el.stHint.textContent =
    sx === sy
      ? `Equal scales → circle of radius ${fmt(sx)}.`
      : `Reach ${fmt(sx)} along x, ${fmt(sy)} along y.`;

  paintCanvas(el.rotIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1, 0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0, 1)");
  });
  paintCanvas(el.rotOut, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(ROT_A, Math.cos(t), Math.sin(t)), INK);
    const [a10, a11] = applyMat2(ROT_A, 1, 0);
    const [a20, a21] = applyMat2(ROT_A, 0, 1);
    drawArrow(ctx, cx, cy, scale, a10, a11, ACCENT2, "R(1,0)");
    drawArrow(ctx, cx, cy, scale, a20, a21, ACCENT, "R(0,1)");
  });

  const stScale = Math.max(2.4, Math.max(sx, sy) * 1.25, 1.2);
  paintCanvas(el.stIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1, 0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0, 1)");
  });
  paintCanvas(el.stOut, stScale, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(STRETCH_A, Math.cos(t), Math.sin(t)), INK);
    if (sx > 1e-6) drawArrow(ctx, cx, cy, scale, sx, 0, ACCENT2, `(${fmt(sx)}, 0)`);
    if (sy > 1e-6) drawArrow(ctx, cx, cy, scale, 0, sy, ACCENT, `(0, ${fmt(sy)})`);
  });
}

function paintMovie(f: EllipseFrame): void {
  const r = f.outScale;
  const [s1, s2] = f.sigma;
  const v1x = get(f.V, 0, 0);
  const v1y = get(f.V, 1, 0);
  const v2x = get(f.V, 0, 1);
  const v2y = get(f.V, 1, 1);

  paintCanvas(el.mv0, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, v1x, v1y, ACCENT2, "v₁");
    drawArrow(ctx, cx, cy, scale, v2x, v2y, ACCENT, "v₂");
  });

  // Vᵀ sends vⱼ onto the coordinate axes — that is the visible rotation.
  const [w1x, w1y] = applyMat2(f.Vt, v1x, v1y);
  const [w2x, w2y] = applyMat2(f.Vt, v2x, v2y);
  paintCanvas(el.mv1, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.Vt, Math.cos(t), Math.sin(t)), INK);
    drawArrow(ctx, cx, cy, scale, w1x, w1y, ACCENT2, "Vᵀv₁");
    drawArrow(ctx, cx, cy, scale, w2x, w2y, ACCENT, "Vᵀv₂");
  });

  const [p1x, p1y] = applyMat2(f.S, w1x, w1y); // ≈ (σ₁, 0)
  const [p2x, p2y] = applyMat2(f.S, w2x, w2y); // ≈ (0, σ₂)
  paintCanvas(el.mv2, r, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.SV, Math.cos(t), Math.sin(t)), INK);
    if (s1 > 1e-6) drawArrow(ctx, cx, cy, scale, p1x, p1y, ACCENT2, "σ₁");
    if (s2 > 1e-6) drawArrow(ctx, cx, cy, scale, p2x, p2y, ACCENT, "σ₂");
  });

  const [q1x, q1y] = applyMat2(f.U, p1x, p1y); // σ₁ u₁
  const [q2x, q2y] = applyMat2(f.U, p2x, p2y); // σ₂ u₂
  paintCanvas(el.mv3, r, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.A, Math.cos(t), Math.sin(t)), INK);
    if (s1 > 1e-6) drawArrow(ctx, cx, cy, scale, q1x, q1y, ACCENT2, "σ₁u₁");
    if (s2 > 1e-6) drawArrow(ctx, cx, cy, scale, q2x, q2y, ACCENT, "σ₂u₂");
  });
}

function syncAnalysisLabels(): void {
  el.a11Val.textContent = fmt(Number(el.a11.value));
  el.a12Val.textContent = fmt(Number(el.a12.value));
  el.a21Val.textContent = fmt(Number(el.a21.value));
  el.a22Val.textContent = fmt(Number(el.a22.value));
}

function redrawAnalysis(): void {
  syncAnalysisLabels();
  const f = rebuildAnalysis();
  const A = f.A;
  el.analysisA.textContent =
    `A = [[${fmt(get(A, 0, 0))}, ${fmt(get(A, 0, 1))}],  [${fmt(get(A, 1, 0))}, ${fmt(get(A, 1, 1))}]]`;
  el.stretchReadout.textContent =
    `Longest stretch σ₁ = ${fmt(f.sigma[0])} · shortest stretch σ₂ = ${fmt(f.sigma[1])} ` +
    `· |det A| = ${fmt(Math.abs(get(A, 0, 0) * get(A, 1, 1) - get(A, 0, 1) * get(A, 1, 0)))}`;
  paintMovie(f);
  void window.MathJax?.typesetPromise?.([app]);
}

function redrawSynth(): void {
  const factors = readSynth();
  el.s1Val.textContent = fmt(factors.s1);
  el.s2Val.textContent = fmt(factors.s2);
  el.thVVal.textContent = `${Math.round(factors.thVDeg)}°`;
  el.thUVal.textContent = `${Math.round(factors.thUDeg)}°`;
  const f = rebuildSynth();
  const A = f.A;
  el.synthA.textContent =
    `A = [[${fmt(get(A, 0, 0))}, ${fmt(get(A, 0, 1))}],  [${fmt(get(A, 1, 0))}, ${fmt(get(A, 1, 1))}]]`;
  el.synthReadout.textContent =
    `σ₁ = ${fmt(f.sigma[0])} · σ₂ = ${fmt(f.sigma[1])} · |det A| = σ₁σ₂ = ${fmt(f.sigma[0] * f.sigma[1])}`;
}

for (const input of [el.a11, el.a12, el.a21, el.a22]) {
  input.addEventListener("input", redrawAnalysis);
}

el.rotDeg.addEventListener("input", paintPieces);
el.sx.addEventListener("input", paintPieces);
el.sy.addEventListener("input", paintPieces);

app.querySelectorAll<HTMLButtonElement>("[data-amat]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = A_PRESETS[btn.dataset.amat ?? ""];
    if (!p) return;
    setA(...p);
    redrawAnalysis();
  });
});

function onSigmaInput(which: "s1" | "s2"): void {
  const s1 = Number(el.s1.value);
  const s2 = Number(el.s2.value);
  if (which === "s1" && s1 < s2) el.s2.value = el.s1.value;
  if (which === "s2" && s2 > s1) el.s1.value = el.s2.value;
  redrawSynth();
}

el.s1.addEventListener("input", () => onSigmaInput("s1"));
el.s2.addEventListener("input", () => onSigmaInput("s2"));
el.thV.addEventListener("input", redrawSynth);
el.thU.addEventListener("input", redrawSynth);

app.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = SYNTH_PRESETS[btn.dataset.preset ?? ""];
    if (!p) return;
    setSynth(p);
    redrawSynth();
  });
});

paintPieces();
redrawAnalysis();
redrawSynth();

let probe = 0.4;
function tick(): void {
  probe += 0.018;
  const af = analysisFrame ?? rebuildAnalysis();
  const sf = synthFrame ?? rebuildSynth();
  paintPair(el.anIn, el.anOut, af, probe, {
    v1: "v₁",
    v2: "v₂",
    u1: "σ₁u₁",
    u2: "σ₂u₂",
  });
  paintPair(el.syIn, el.syOut, sf, probe, {
    v1: "v₁",
    v2: "v₂",
    u1: "σ₁u₁",
    u2: "σ₂u₂",
  });
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

window.addEventListener("resize", () => {
  paintPieces();
  const af = analysisFrame ?? rebuildAnalysis();
  paintMovie(af);
  paintPair(el.anIn, el.anOut, af, probe, {
    v1: "v₁",
    v2: "v₂",
    u1: "σ₁u₁",
    u2: "σ₂u₂",
  });
  const sf = synthFrame ?? rebuildSynth();
  paintPair(el.syIn, el.syOut, sf, probe, {
    v1: "v₁",
    v2: "v₂",
    u1: "σ₁u₁",
    u2: "σ₂u₂",
  });
});
