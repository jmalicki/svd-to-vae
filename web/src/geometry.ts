import "./style.css";
import { chapterNav } from "./chapterNav";
import {
  TILTED_EXAMPLE_A,
  applyMat2,
  frameFromFactors,
  frameFromMatrix,
  type AngleFactors,
  type EllipseFrame,
} from "./svdGeometry2d";
import { fromNested, get, type Matrix } from "./matrix";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

/** Fixed tilted example: σ ≠ diagonal entries. */
const EXAMPLE_A = TILTED_EXAMPLE_A;

app.innerHTML = `
  <header>
    ${chapterNav({ current: 1, next: { href: "./truncate.html", label: "Next →" } })}
    <h1>A matrix turns a circle into an ellipse</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      Pick a $2\\times 2$ matrix $A$. Send every vector of length $1$ through $Ax$.
      The tips no longer form a circle — they form an <strong>ellipse</strong>.
      This page unpacks that one fact.
    </p>
  </header>

  <section class="theory" aria-label="Setup">
    <h2>Setup</h2>
    <ol class="theory-steps">
      <li>
        <p>
          <strong>Before / after.</strong>
          Left: unit circle (every tip has length $1$). Right: the same tips after multiplying by $A$.
        </p>
      </li>
      <li>
        <p>
          <strong>Some directions stretch more than others.</strong>
          The longest and shortest stretches on that ellipse will get names in a moment —
          after you can see them.
        </p>
      </li>
    </ol>
  </section>

  <section class="worked-example" aria-label="Worked example">
    <h2>One concrete example (tilted)</h2>
    <p>
      Here is a matrix that is <em>not</em> diagonal — so the ellipse will not line up with the axes,
      and the stretch lengths will <em>not</em> equal the numbers on the diagonal of $A$.
    </p>
    <div class="math">
      $$A = \\begin{bmatrix} 1.5 & 1.0 \\\\ 0.2 & 1.2 \\end{bmatrix}$$
    </div>
    <ul class="example-bullets">
      <li>$(1,0)$ goes to $A(1,0)=(1.5,\\,0.2)$ — length $\\sqrt{1.5^{2}+0.2^{2}}\\approx 1.51$.</li>
      <li>$(0,1)$ goes to $A(0,1)=(1.0,\\,1.2)$ — length $\\sqrt{1^{2}+1.2^{2}}\\approx 1.56$.</li>
      <li>Those are just two directions. Sweep the whole circle → the ellipse on the right.</li>
    </ul>
    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Before: unit circle</h2>
        <canvas id="exIn" width="320" height="320" aria-label="Example unit circle"></canvas>
        <p class="hint">Orange $(1,0)$. Blue $(0,1)$.</p>
      </div>
      <div class="panel">
        <h2>After: multiply by $A$</h2>
        <canvas id="exOut" width="320" height="320" aria-label="Example ellipse"></canvas>
        <p class="hint">Same two arrows after $A$. The whole circle becomes a tilted ellipse.</p>
      </div>
    </div>
    <p class="example-takeaway" id="exampleStretch"></p>
  </section>

  <section class="demo-block" aria-label="Analysis playground">
    <h2>Try your own matrix</h2>
    <p class="demo-intro">
      Edit the four entries of $A$. Watch the ellipse, and read off the longest and shortest
      stretches over the circle. Those two lengths are the <strong>singular values</strong>
      $\\sigma_1 \\ge \\sigma_2$.
    </p>

    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">$a_{11}$ <strong id="a11Val">1.20</strong></span>
          <input id="a11" type="range" min="-2.5" max="2.5" step="0.05" value="1.2" />
        </label>
        <label class="slider">
          <span class="slider-label">$a_{12}$ <strong id="a12Val">0.90</strong></span>
          <input id="a12" type="range" min="-2.5" max="2.5" step="0.05" value="0.9" />
        </label>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">$a_{21}$ <strong id="a21Val">0.40</strong></span>
          <input id="a21" type="range" min="-2.5" max="2.5" step="0.05" value="0.4" />
        </label>
        <label class="slider">
          <span class="slider-label">$a_{22}$ <strong id="a22Val">1.50</strong></span>
          <input id="a22" type="range" min="-2.5" max="2.5" step="0.05" value="1.5" />
        </label>
      </div>
      <div class="control-actions">
        <div class="btns">
          <button type="button" data-amat="default">Default</button>
          <button type="button" data-amat="example" class="secondary">Match example</button>
          <button type="button" data-amat="diag" class="secondary">Diagonal</button>
          <button type="button" data-amat="flat" class="secondary">Nearly flat</button>
        </div>
        <p class="help">Compare diagonal (axes lined up) vs tilted. Nearly flat → one stretch near zero.</p>
      </div>
    </div>

    <p class="formula" id="analysisA" aria-live="polite"></p>
    <p class="status" id="stretchReadout" aria-live="polite"></p>

    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Before: unit circle</h2>
        <canvas id="anIn" width="360" height="360" aria-label="Analysis unit circle"></canvas>
        <p class="hint">
          Orange / blue: the two special input directions $v_1$, $v_2$ (right singular vectors).
          Moving dot: one $x$ on the circle.
        </p>
      </div>
      <div class="panel">
        <h2>After: multiply by $A$</h2>
        <canvas id="anOut" width="360" height="360" aria-label="Analysis ellipse"></canvas>
        <p class="hint">
          Axis lengths are $\\sigma_1$ and $\\sigma_2$. Arrows $\\sigma_j u_j = A v_j$.
          Moving dot: $Ax$.
        </p>
      </div>
    </div>
  </section>

  <section class="movie-block" aria-label="Rotate stretch rotate">
    <h2>How $A$ does it: rotate, stretch, rotate</h2>
    <p class="demo-intro">
      Any $2\\times 2$ $A$ can be written as three easy steps applied right-to-left.
      The pictures below use the SVD of <em>your</em> matrix from the section above.
    </p>
    <div class="movie-row" id="movieRow">
      <div class="panel movie-panel">
        <h2>1 · Start</h2>
        <canvas id="mv0" width="200" height="200" aria-label="Unit circle"></canvas>
        <p class="hint">Unit circle.</p>
      </div>
      <div class="panel movie-panel">
        <h2>2 · Re-aim ($V^{\\top}$)</h2>
        <canvas id="mv1" width="200" height="200" aria-label="After V transpose"></canvas>
        <p class="hint">Rotate so special directions sit on the axes.</p>
      </div>
      <div class="panel movie-panel">
        <h2>3 · Stretch ($\\Sigma$)</h2>
        <canvas id="mv2" width="200" height="200" aria-label="After Sigma"></canvas>
        <p class="hint">Stretch axes by $\\sigma_1$, $\\sigma_2$.</p>
      </div>
      <div class="panel movie-panel">
        <h2>4 · Re-aim ($U$)</h2>
        <canvas id="mv3" width="200" height="200" aria-label="After U"></canvas>
        <p class="hint">Rotate to the final ellipse — same as $Ax$.</p>
      </div>
    </div>
    <div class="math">
      $$A = U\\,\\mathrm{diag}(\\sigma_1,\\sigma_2)\\,V^{\\top}$$
    </div>
    <p class="example-takeaway">
      In words: rotate → stretch by $\\sigma_1$ and $\\sigma_2$ → rotate again.
      That recipe is the
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD).
    </p>
  </section>

  <section class="demo-block" aria-label="Synthesis playground">
    <h2>Rebuild $A$ from the pieces</h2>
    <p class="demo-intro">
      Now scrub the stretches and the two rotation angles yourself.
      $A$ is rebuilt as $U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$.
    </p>

    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Long stretch $\\sigma_1$ <strong id="s1Val">2.20</strong></span>
          <input id="s1" type="range" min="0" max="3" step="0.05" value="2.2" />
        </label>
        <p class="help">Long axis of the ellipse.</p>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Short stretch $\\sigma_2$ <strong id="s2Val">0.80</strong></span>
          <input id="s2" type="range" min="0" max="3" step="0.05" value="0.8" />
        </label>
        <p class="help">Short axis. Zero → everything collapses to a line.</p>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Input twist ($V$) <strong id="thVVal">25°</strong></span>
          <input id="thV" type="range" min="-90" max="90" step="1" value="25" />
        </label>
        <p class="help">Which circle directions get the long vs short stretch.</p>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Output twist ($U$) <strong id="thUVal">40°</strong></span>
          <input id="thU" type="range" min="-90" max="90" step="1" value="40" />
        </label>
        <p class="help">Which way the finished ellipse points.</p>
      </div>
      <div class="control-actions">
        <div class="btns">
          <button type="button" data-preset="stretch">Stretch</button>
          <button type="button" data-preset="circle" class="secondary">Same stretches</button>
          <button type="button" data-preset="flat" class="secondary">Flatten</button>
          <button type="button" data-preset="tilt" class="secondary">Tilted</button>
        </div>
        <p class="help">Presets set $\\sigma$ and both angles.</p>
      </div>
    </div>

    <p class="formula" id="synthA" aria-live="polite"></p>
    <p class="status" id="synthReadout" aria-live="polite"></p>

    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Before: unit circle</h2>
        <canvas id="syIn" width="360" height="360" aria-label="Synthesis unit circle"></canvas>
      </div>
      <div class="panel">
        <h2>After: multiply by rebuilt $A$</h2>
        <canvas id="syOut" width="360" height="360" aria-label="Synthesis ellipse"></canvas>
      </div>
    </div>
  </section>

  <section class="appendix" id="appendix" aria-label="Takeaways">
    <h2>Takeaways</h2>
    <ul class="example-bullets">
      <li>$A$ sends the unit circle to an ellipse.</li>
      <li>$\\sigma_1$ and $\\sigma_2$ are that ellipse’s long and short axis lengths — not (in general) the diagonal of $A$.</li>
      <li>Any $A$ is rotate → stretch → rotate: $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$.</li>
    </ul>
    <p>
      <a href="./truncate.html">Next →</a>
      keep only the biggest stretches and throw the small ones away.
    </p>
  </section>
`;

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
  exampleStretch: app.querySelector<HTMLElement>("#exampleStretch")!,
  exIn: app.querySelector<HTMLCanvasElement>("#exIn")!,
  exOut: app.querySelector<HTMLCanvasElement>("#exOut")!,
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
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const css = canvas.clientWidth || canvas.width || 200;
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

function paintExample(): void {
  const f = frameFromMatrix(EXAMPLE_A);
  paintCanvas(el.exIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1, 0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0, 1)");
  });
  paintCanvas(el.exOut, f.outScale, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(EXAMPLE_A, Math.cos(t), Math.sin(t)), INK);
    const [a10, a11] = applyMat2(EXAMPLE_A, 1, 0);
    const [a20, a21] = applyMat2(EXAMPLE_A, 0, 1);
    drawArrow(ctx, cx, cy, scale, a10, a11, ACCENT2, "A(1,0)");
    drawArrow(ctx, cx, cy, scale, a20, a21, ACCENT, "A(0,1)");
    drawArrow(ctx, cx, cy, scale, get(f.U, 0, 0) * f.sigma[0], get(f.U, 1, 0) * f.sigma[0], "#666", "long axis");
  });
  el.exampleStretch.textContent =
    `Longest stretch on this ellipse: σ₁ ≈ ${fmt(f.sigma[0])}. ` +
    `Shortest: σ₂ ≈ ${fmt(f.sigma[1])}. ` +
    `Compare to the diagonal entries 1.5 and 1.2 — not the same.`;
}

function paintMovie(f: EllipseFrame): void {
  const r = f.outScale;
  paintCanvas(el.mv0, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
  });
  paintCanvas(el.mv1, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.Vt, Math.cos(t), Math.sin(t)), INK);
  });
  paintCanvas(el.mv2, r, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.SV, Math.cos(t), Math.sin(t)), INK);
  });
  paintCanvas(el.mv3, r, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (t) => applyMat2(f.A, Math.cos(t), Math.sin(t)), INK);
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

paintExample();
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
  paintExample();
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
