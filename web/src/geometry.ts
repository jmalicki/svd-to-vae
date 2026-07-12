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
      A $2\\times 2$ matrix $A$ is a linear map of the plane: it takes every vector $x$ to $Ax$.
      The
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD) writes that map as stretch + rotations:
      $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$.
      Scrub the singular values and angles below — the matrix and the ellipse are built from them.
    </p>
  </header>

  <section class="theory" aria-label="SVD geometry">
    <h2>What you are looking at</h2>
    <ol class="theory-steps">
      <li>
        <p>
          Start with the <strong>unit circle</strong>: every vector $x$ with length $1$.
          Apply $A$. The image $\\{Ax : \\|x\\|=1\\}$ is an <strong>ellipse</strong>
          (or a line segment if one singular value is zero).
        </p>
      </li>
      <li>
        <p>
          The <strong>singular values</strong> $\\sigma_1 \\ge \\sigma_2 \\ge 0$ are the semi-axis lengths —
          how far $A$ stretches along its two principal directions. Change them and the ellipse
          (and the matrix entries) change immediately.
        </p>
        <div class="math">
          $$A = U\\,\\mathrm{diag}(\\sigma_1,\\sigma_2)\\,V^{\\top}$$
        </div>
      </li>
      <li>
        <p>
          Angle of $V$ aims the input axes on the circle; angle of $U$ aims the output axes on the
          ellipse. $U$ and $V$ only rotate; all the stretching lives in $\\sigma$.
        </p>
      </li>
    </ol>
  </section>

  <div class="controls">
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">$\\sigma_1$ (major) <strong id="s1Val">2.20</strong></span>
        <input id="s1" type="range" min="0" max="3" step="0.05" value="2.2" />
      </label>
      <p class="help">Longer ellipse axis. Larger $\\sigma_1$ stretches harder along $u_1$.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">$\\sigma_2$ (minor) <strong id="s2Val">0.80</strong></span>
        <input id="s2" type="range" min="0" max="3" step="0.05" value="0.8" />
      </label>
      <p class="help">Shorter axis. Set to $0$ to collapse the ellipse to a line (rank-$1$ matrix).</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">$V$ angle <strong id="thVVal">25°</strong></span>
        <input id="thV" type="range" min="-90" max="90" step="1" value="25" />
      </label>
      <p class="help">Rotates the input directions $v_1,v_2$ on the unit circle.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">$U$ angle <strong id="thUVal">40°</strong></span>
        <input id="thU" type="range" min="-90" max="90" step="1" value="40" />
      </label>
      <p class="help">Rotates the output axes of the ellipse (columns of $U$).</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button type="button" data-preset="stretch">Stretch</button>
        <button type="button" data-preset="circle" class="secondary">Isotropic</button>
        <button type="button" data-preset="flat" class="secondary">Rank-1</button>
        <button type="button" data-preset="tilt" class="secondary">Tilted</button>
      </div>
      <p class="help">Presets set $\\sigma$ and the two angles; drag any slider to explore further.</p>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>Matrix $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$</h2>
      <canvas id="Aheat" width="160" height="160" aria-label="Heatmap of matrix A"></canvas>
      <p class="formula" id="matrixReadout" aria-live="polite"></p>
    </div>
    <div class="panel sigma-bars-panel">
      <h2>Singular values</h2>
      <canvas id="sigmaBars" width="200" height="120" aria-label="Bar chart of singular values"></canvas>
      <p class="status" id="sigmaReadout" aria-live="polite"></p>
    </div>
  </div>

  <div class="grid-2 ellipse-pair">
    <div class="panel">
      <h2>Unit circle · input ($V$)</h2>
      <canvas id="inPlane" width="360" height="360" aria-label="Unit circle with right singular vectors"></canvas>
      <p class="hint">
        Orange / blue arrows: columns of $V$. The moving dot is a unit vector $x$;
        watch where $Ax$ lands on the right.
      </p>
    </div>
    <div class="panel">
      <h2>Image under $A$ · ellipse ($U$, $\\sigma$)</h2>
      <canvas id="outPlane" width="360" height="360" aria-label="Ellipse with left singular vectors scaled by sigma"></canvas>
      <p class="hint">
        Axis lengths are exactly $\\sigma_1$ and $\\sigma_2$. Arrows are $\\sigma_j u_j = A v_j$.
      </p>
    </div>
  </div>

  <section class="appendix" id="appendix" aria-label="Appendix">
    <h2>Appendix: reading the factors</h2>
    <h3>Right singular vectors $V$</h3>
    <p>
      Orthonormal columns $v_1, v_2$ on the input side. They pick the two special directions
      on the unit circle that map to the ellipse axes. Here a single angle builds
      $V = \\begin{bmatrix}\\cos\\theta_V & -\\sin\\theta_V\\\\ \\sin\\theta_V & \\cos\\theta_V\\end{bmatrix}$.
    </p>
    <h3>Singular values $\\sigma$</h3>
    <p>
      Nonnegative stretch factors. If $\\sigma_2 = 0$, the ellipse collapses to a line segment and
      $A$ has rank at most $1$. If $\\sigma_1 = \\sigma_2$, every direction stretches the same —
      the “ellipse” is a circle (possibly rotated by $U$ relative to $V$).
    </p>
    <h3>Left singular vectors $U$</h3>
    <p>
      Orthonormal columns $u_1, u_2$ on the output side. Together: $A v_j = \\sigma_j u_j$.
      Multiplying $U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$ rebuilds every entry of $A$.
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

function paintPlanes(probeAngle: number): void {
  const f = frame ?? rebuildFrame();
  const { A, U, V, sigma, outScale } = f;

  const paint = (
    canvas: HTMLCanvasElement,
    worldR: number,
    drawContent: (ctx: CanvasRenderingContext2D, cx: number, cy: number, scale: number) => void,
  ) => {
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
  };

  paint(el.inPlane, 1.35, (ctx, cx, cy, scale) => {
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

  paint(el.outPlane, outScale, (ctx, cx, cy, scale) => {
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

let probe = 0.4;
function tick(): void {
  probe += 0.018;
  paintPlanes(probe);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

window.addEventListener("resize", () => paintPlanes(probe));
