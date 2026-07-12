import "./style.css";
import { chapterNav } from "./chapterNav";
import {
  fromNested,
  get,
  matmul,
  transpose,
  type Matrix,
} from "./matrix";
import { drawHeatmap } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    ${chapterNav({ current: 1, next: { href: "./truncate.html", label: "Next →" } })}
    <h1>A matrix turns a circle into an ellipse</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      Take every vector of length 1 — their tips form a circle — and multiply each by a
      $2\\times 2$ matrix $A$. The tips no longer form a circle. They form an
      <strong>ellipse</strong>. The rest of this page is that one fact, slowly unpacked.
    </p>
  </header>

  <section class="theory" aria-label="Setup">
    <h2>Before you play</h2>
    <ol class="theory-steps">
      <li>
        <p>
          <strong>A matrix moves every vector.</strong>
          For a vector $x$, the product $Ax$ is another vector.
          Left picture = before. Right picture = after.
        </p>
      </li>
      <li>
        <p>
          <strong>Some directions stretch more than others.</strong>
          The longest and shortest stretches on that ellipse are special numbers called
          <em>singular values</em> (written $\\sigma_1$ and $\\sigma_2$, “sigma”).
          Start with one fixed matrix below, then scrub your own.
        </p>
      </li>
    </ol>
  </section>

  <section class="worked-example" aria-label="Worked example">
    <h2>One concrete example</h2>
    <p>
      Here’s a simple matrix: stretch sideways by $2$, squash up–down by $\\tfrac{1}{2}$.
      Off-diagonal entries are zero, so the axes stay lined up with the coordinate grid.
    </p>
    <div class="math">
      $$A = \\begin{bmatrix} 2 & 0 \\\\ 0 & 0.5 \\end{bmatrix}$$
    </div>
    <ul class="example-bullets">
      <li>Right unit vector $x=(1,0)$ becomes $Ax=(2,0)$ — twice as long.</li>
      <li>Up unit vector $x=(0,1)$ becomes $Ax=(0,0.5)$ — half as long.</li>
      <li>Do that for every direction on the circle → the ellipse on the right.</li>
    </ul>
    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Before: unit circle</h2>
        <canvas id="exIn" width="360" height="360" aria-label="Example unit circle"></canvas>
        <p class="hint">Orange arrow $(1,0)$. Blue arrow $(0,1)$.</p>
      </div>
      <div class="panel">
        <h2>After: multiply by $A$</h2>
        <canvas id="exOut" width="360" height="360" aria-label="Example ellipse"></canvas>
        <p class="hint">Same arrows after $A$: lengths $2$ and $0.5$. Those lengths are $\\sigma_1$ and $\\sigma_2$.</p>
      </div>
    </div>
    <p class="example-takeaway">
      So for this $A$, the singular values are just $\\sigma_1=2$ and $\\sigma_2=0.5$ —
      the ellipse’s long and short axis lengths. The
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD) finds those stretches (and two rotations) for <em>any</em> $2\\times 2$ matrix,
      even when the ellipse is tilted.
    </p>
  </section>

  <section class="demo-block" aria-label="Interactive demo">
    <h2>Try it yourself</h2>
    <p class="demo-intro">
      Change the two stretch amounts and two twist angles.
      The matrix $A$ is rebuilt as rotate → stretch → rotate again.
      Watch the heatmap and the ellipse update together.
    </p>

  <div class="controls">
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Long stretch $\\sigma_1$ <strong id="s1Val">2.20</strong></span>
        <input id="s1" type="range" min="0" max="3" step="0.05" value="2.2" />
      </label>
      <p class="help">How far the ellipse reaches along its long axis.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Short stretch $\\sigma_2$ <strong id="s2Val">0.80</strong></span>
        <input id="s2" type="range" min="0" max="3" step="0.05" value="0.8" />
      </label>
      <p class="help">How far along the short axis. Zero → flat line (the matrix “loses a dimension”).</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Input twist ($V$) <strong id="thVVal">25°</strong></span>
        <input id="thV" type="range" min="-90" max="90" step="1" value="25" />
      </label>
      <p class="help">Which directions on the circle get the long vs short stretch.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Output twist ($U$) <strong id="thUVal">40°</strong></span>
        <input id="thU" type="range" min="-90" max="90" step="1" value="40" />
      </label>
      <p class="help">Which way the stretched ellipse points in the plane.</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button type="button" data-preset="stretch">Stretch</button>
        <button type="button" data-preset="circle" class="secondary">Same stretches</button>
        <button type="button" data-preset="flat" class="secondary">Flatten</button>
        <button type="button" data-preset="tilt" class="secondary">Tilted</button>
      </div>
      <p class="help">Try a preset, then drag. Watch the matrix numbers change as you move $\\sigma$ and the angles.</p>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>The matrix $A$ (built from the pieces above)</h2>
      <canvas id="Aheat" width="160" height="160" aria-label="Heatmap of matrix A"></canvas>
      <p class="formula" id="matrixReadout" aria-live="polite"></p>
    </div>
    <div class="panel sigma-bars-panel">
      <h2>Stretch amounts $\\sigma_1$, $\\sigma_2$</h2>
      <canvas id="sigmaBars" width="200" height="120" aria-label="Bar chart of singular values"></canvas>
      <p class="status" id="sigmaReadout" aria-live="polite"></p>
    </div>
  </div>

  <div class="grid-2 ellipse-pair">
    <div class="panel">
      <h2>Before: unit circle</h2>
      <canvas id="inPlane" width="360" height="360" aria-label="Unit circle with right singular vectors"></canvas>
      <p class="hint">
        Orange and blue arrows are the two special input directions ($v_1$, $v_2$).
        The moving dot is one vector $x$ riding around the circle.
      </p>
    </div>
    <div class="panel">
      <h2>After: multiply by $A$</h2>
      <canvas id="outPlane" width="360" height="360" aria-label="Ellipse with left singular vectors scaled by sigma"></canvas>
      <p class="hint">
        Same vectors after $A$. Axis lengths are $\\sigma_1$ and $\\sigma_2$.
        The moving dot is $Ax$ — where that input landed.
      </p>
    </div>
  </div>
  </section>

  <section class="appendix" id="appendix" aria-label="Appendix">
    <h2>Appendix: names for the pieces</h2>
    <h3>Singular values $\\sigma$</h3>
    <p>
      Just the two stretch lengths, with $\\sigma_1 \\ge \\sigma_2 \\ge 0$ by convention.
      If $\\sigma_2 = 0$, every output lies on a line — people say the matrix has
      <em>rank 1</em>. If $\\sigma_1 = \\sigma_2$, every direction stretches the same, so you get a circle.
    </p>
    <h3>Matrices $U$ and $V$</h3>
    <p>
      Each is a pure rotation (built from one angle). Columns of $V$ are the special input
      directions; columns of $U$ are the matching output directions.
      The stretch turns $v_j$ into $\\sigma_j u_j$: same idea as “that arrow got longer by $\\sigma_j$.”
    </p>
    <h3>Putting $A$ back together</h3>
    <p>
      Multiply rotate → stretch → rotate and you recover every entry of $A$:
    </p>
    <div class="math">
      $$A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$$
    </div>
    <p>
      That factorization is the SVD. Next page: keep only the biggest stretches and throw the
      small ones away.
    </p>
  </section>
`;

type Factors = { s1: number; s2: number; thVDeg: number; thUDeg: number };

const PRESETS: Record<string, Factors> = {
  stretch: { s1: 2.4, s2: 0.7, thVDeg: 0, thUDeg: 0 },
  circle: { s1: 1.4, s2: 1.4, thVDeg: 0, thUDeg: 35 },
  flat: { s1: 2.2, s2: 0, thVDeg: 20, thUDeg: 55 },
  tilt: { s1: 2.0, s2: 0.9, thVDeg: 35, thUDeg: -20 },
};

const el = {
  s1: app.querySelector<HTMLInputElement>("#s1")!,
  s2: app.querySelector<HTMLInputElement>("#s2")!,
  thV: app.querySelector<HTMLInputElement>("#thV")!,
  thU: app.querySelector<HTMLInputElement>("#thU")!,
  s1Val: app.querySelector<HTMLElement>("#s1Val")!,
  s2Val: app.querySelector<HTMLElement>("#s2Val")!,
  thVVal: app.querySelector<HTMLElement>("#thVVal")!,
  thUVal: app.querySelector<HTMLElement>("#thUVal")!,
  matrixReadout: app.querySelector<HTMLElement>("#matrixReadout")!,
  sigmaReadout: app.querySelector<HTMLElement>("#sigmaReadout")!,
  Aheat: app.querySelector<HTMLCanvasElement>("#Aheat")!,
  sigmaBars: app.querySelector<HTMLCanvasElement>("#sigmaBars")!,
  exIn: app.querySelector<HTMLCanvasElement>("#exIn")!,
  exOut: app.querySelector<HTMLCanvasElement>("#exOut")!,
  inPlane: app.querySelector<HTMLCanvasElement>("#inPlane")!,
  outPlane: app.querySelector<HTMLCanvasElement>("#outPlane")!,
};

const ACCENT = "#0072b2";
const ACCENT2 = "#e69f00";
const INK = "#2d2d2d";
const MUTED = "rgba(45,45,45,0.35)";

function fmt(v: number): string {
  return v.toFixed(2);
}

function rad(d: number): number {
  return (d * Math.PI) / 180;
}

function readFactors(): Factors {
  return {
    s1: Number(el.s1.value),
    s2: Number(el.s2.value),
    thVDeg: Number(el.thV.value),
    thUDeg: Number(el.thU.value),
  };
}

function rot2(theta: number): Matrix {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return fromNested([
    [c, -s],
    [s, c],
  ]);
}

function applyA(A: Matrix, x: number, y: number): [number, number] {
  return [
    get(A, 0, 0) * x + get(A, 0, 1) * y,
    get(A, 1, 0) * x + get(A, 1, 1) * y,
  ];
}

function setFactors(f: Factors): void {
  el.s1.value = String(f.s1);
  el.s2.value = String(f.s2);
  el.thV.value = String(f.thVDeg);
  el.thU.value = String(f.thUDeg);
}

function syncLabels(f: Factors): void {
  el.s1Val.textContent = fmt(Number(el.s1.value));
  el.s2Val.textContent = fmt(Number(el.s2.value));
  el.thVVal.textContent = `${Math.round(f.thVDeg)}°`;
  el.thUVal.textContent = `${Math.round(f.thUDeg)}°`;
}

type Frame = {
  A: Matrix;
  U: Matrix;
  V: Matrix;
  sigma: [number, number];
  outScale: number;
};

let frame: Frame | null = null;

function rebuildFrame(): Frame {
  const f = readFactors();
  const U = rot2(rad(f.thUDeg));
  const V = rot2(rad(f.thVDeg));
  const S = fromNested([
    [f.s1, 0],
    [0, f.s2],
  ]);
  const A = matmul(matmul(U, S), transpose(V));
  const outScale = Math.max(2.4, f.s1 * 1.25, 1.2);
  frame = { A, U, V, sigma: [f.s1, f.s2], outScale };
  return frame;
}

function drawSigmaBars(canvas: HTMLCanvasElement, s1: number, s2: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const peak = Math.max(s1, s2, 1e-9);
  const gap = 16;
  const barW = (w - gap * 3) / 2;
  const colors = [ACCENT2, ACCENT];
  const vals = [s1, s2];
  const labels = ["σ₁", "σ₂"];
  for (let i = 0; i < 2; i++) {
    const bh = (vals[i] / peak) * (h - 28);
    const x = gap + i * (barW + gap);
    const y = h - 18 - bh;
    ctx.fillStyle = colors[i];
    ctx.fillRect(x, y, barW, bh);
    ctx.fillStyle = INK;
    ctx.font = "600 12px IBM Plex Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(labels[i], x + barW / 2, h - 4);
  }
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
  const head = 10;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
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
  ctx.font = "600 13px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, x1 + 8 * ux, y1 + 8 * uy);
}

function drawCircleOrEllipse(
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
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const css = canvas.clientWidth || 360;
  canvas.width = Math.round(css * dpr);
  canvas.height = Math.round(css * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = css;
  ctx.clearRect(0, 0, w, w);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, w);
  const cx = w / 2;
  const cy = w / 2;
  const scale = (0.42 * w) / worldR;
  drawAxes(ctx, cx, cy, scale, worldR);
  drawContent(ctx, cx, cy, scale);
}

const EXAMPLE_A = fromNested([
  [2, 0],
  [0, 0.5],
]);
const EXAMPLE_OUT_SCALE = 2.6;

function paintExample(): void {
  paintCanvas(el.exIn, 1.35, (ctx, cx, cy, scale) => {
    drawCircleOrEllipse(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1, 0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0, 1)");
  });
  paintCanvas(el.exOut, EXAMPLE_OUT_SCALE, (ctx, cx, cy, scale) => {
    drawCircleOrEllipse(
      ctx,
      cx,
      cy,
      scale,
      (t) => applyA(EXAMPLE_A, Math.cos(t), Math.sin(t)),
      INK,
    );
    drawArrow(ctx, cx, cy, scale, 2, 0, ACCENT2, "(2, 0)");
    drawArrow(ctx, cx, cy, scale, 0, 0.5, ACCENT, "(0, 0.5)");
  });
}

function paintPlanes(probeAngle: number): void {
  const f = frame ?? rebuildFrame();
  const { A, U, V, sigma, outScale } = f;

  paintCanvas(el.inPlane, 1.35, (ctx, cx, cy, scale) => {
    drawCircleOrEllipse(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, get(V, 0, 0), get(V, 1, 0), ACCENT2, "v₁");
    drawArrow(ctx, cx, cy, scale, get(V, 0, 1), get(V, 1, 1), ACCENT, "v₂");
    const px = Math.cos(probeAngle);
    const py = Math.sin(probeAngle);
    const [qx, qy] = toCanvas(cx, cy, scale, px, py);
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(qx, qy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "500 12px DM Sans, system-ui, sans-serif";
    ctx.fillStyle = "rgba(45,45,45,0.75)";
    ctx.fillText("x", qx + 8, qy - 8);
  });

  paintCanvas(el.outPlane, outScale, (ctx, cx, cy, scale) => {
    drawCircleOrEllipse(
      ctx,
      cx,
      cy,
      scale,
      (t) => applyA(A, Math.cos(t), Math.sin(t)),
      INK,
    );
    const [s1, s2] = sigma;
    drawArrow(ctx, cx, cy, scale, get(U, 0, 0) * s1, get(U, 1, 0) * s1, ACCENT2, "σ₁u₁");
    if (s2 > 1e-6) {
      drawArrow(ctx, cx, cy, scale, get(U, 0, 1) * s2, get(U, 1, 1) * s2, ACCENT, "σ₂u₂");
    }
    const [ax, ay] = applyA(A, Math.cos(probeAngle), Math.sin(probeAngle));
    const [qx, qy] = toCanvas(cx, cy, scale, ax, ay);
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(qx, qy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "500 12px DM Sans, system-ui, sans-serif";
    ctx.fillStyle = "rgba(45,45,45,0.75)";
    ctx.fillText("Ax", qx + 8, qy - 8);
  });
}

function redrawStatic(): void {
  const factors = readFactors();
  syncLabels(factors);
  const f = rebuildFrame();
  const A = f.A;
  drawHeatmap(el.Aheat, A, 3);
  drawSigmaBars(el.sigmaBars, f.sigma[0], f.sigma[1]);
  el.matrixReadout.textContent =
    `[[${fmt(get(A, 0, 0))}, ${fmt(get(A, 0, 1))}],  [${fmt(get(A, 1, 0))}, ${fmt(get(A, 1, 1))}]]`;
  el.sigmaReadout.textContent =
    `σ₁ = ${fmt(f.sigma[0])} · σ₂ = ${fmt(f.sigma[1])} · |det A| = σ₁σ₂ = ${fmt(f.sigma[0] * f.sigma[1])}`;
  void window.MathJax?.typesetPromise?.([app]);
}

function onSigmaInput(which: "s1" | "s2"): void {
  const s1 = Number(el.s1.value);
  const s2 = Number(el.s2.value);
  if (which === "s1" && s1 < s2) el.s2.value = el.s1.value;
  if (which === "s2" && s2 > s1) el.s1.value = el.s2.value;
  redrawStatic();
}

el.s1.addEventListener("input", () => onSigmaInput("s1"));
el.s2.addEventListener("input", () => onSigmaInput("s2"));
el.thV.addEventListener("input", redrawStatic);
el.thU.addEventListener("input", redrawStatic);

app.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = PRESETS[btn.dataset.preset ?? ""];
    if (!p) return;
    setFactors(p);
    redrawStatic();
  });
});

redrawStatic();
paintExample();

let probe = 0.4;
function tick(): void {
  probe += 0.018;
  paintPlanes(probe);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

window.addEventListener("resize", () => {
  paintExample();
  paintPlanes(probe);
});
