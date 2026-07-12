import "./style.css";
import { chapterNav } from "./chapterNav";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
} from "./matrix";
import { drawHeatmap, drawSigmaBars } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    ${chapterNav({
      current: 2,
      prev: { href: "./", label: "← Matrix" },
      next: { href: "./truncate.html", label: "Next →" },
    })}
    <h1>The singular value decomposition</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      Last page: every $2\\times 2$ linear transformation is rotate, stretch, rotate again.
      That factorization has a name — the
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD) — and it works for any size of matrix, not only $2\\times 2$.
    </p>
  </header>

  <section class="theory" aria-label="SVD">
    <h2>What the name means</h2>
    <ol class="theory-steps">
      <li>
        <p>
          Any real matrix $A$ factors as a product of two rotations (or reflections) and a stretch:
        </p>
        <div class="math">
          $$A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$$
        </div>
        <p>
          $V^{\\top}$ rotates the input, $\\mathrm{diag}(\\sigma)$ stretches each axis by a
          nonnegative amount $\\sigma_j$ (the <strong>singular values</strong>), and $U$ rotates
          the result. Same three steps you saw on the unit circle — now written as matrices.
        </p>
      </li>
      <li>
        <p>
          $U$ and $V$ have
          <a href="https://en.wikipedia.org/wiki/Orthonormality" target="_blank" rel="noopener noreferrer">orthonormal</a>
          columns: they change directions without changing lengths.
          The numbers $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge 0$ are how hard $A$ stretches
          along those special directions.
        </p>
      </li>
      <li>
        <p>
          Multiply the factors back together and you recover $A$ exactly
          (up to floating-point noise). Nothing is thrown away yet — this page is only the
          full decomposition.
        </p>
      </li>
    </ol>
  </section>

  <div class="controls">
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Size <em>n</em> <strong id="sizeVal">5</strong></span>
        <input id="size" type="range" min="0" max="10" step="1" value="2" />
      </label>
      <p class="help">Side length of random square <code>A</code> (n×n). The slider is log-spaced (3…32).</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button id="regen" type="button">Regenerate A</button>
      </div>
      <p class="help">Draw a new random Gaussian matrix and recompute the full SVD.</p>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>Matrix A</h2>
      <canvas id="A" width="160" height="160"></canvas>
    </div>
  </div>

  <div class="panel" id="svdCol">
    <h2>Full SVD</h2>
    <div class="factor-row">
      <div class="factor"><span>U</span><canvas id="svdU" width="100" height="100"></canvas></div>
      <div class="factor"><span>σ</span><canvas id="svdS" width="100" height="80"></canvas></div>
      <div class="factor"><span>V</span><canvas id="svdV" width="100" height="100"></canvas></div>
      <div class="factor"><span>U diag(σ) Vᵀ</span><canvas id="svdRecon" width="120" height="120"></canvas></div>
    </div>
    <p class="note" id="svdErr"></p>
  </div>

  <section class="conclusion" id="conclusion" aria-label="Summary">
    <h2>In short</h2>
    <p>
      The SVD is the name for factoring any matrix into rotate–stretch–rotate:
      $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$.
      The singular values $\\sigma$ are the stretch amounts.
    </p>
    <p class="next-chapter">
      <a href="./truncate.html">Keeping only some of the stretches →</a>
    </p>
  </section>
`;

const el = {
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

/** Log-spaced matrix sizes for the n slider (index → n). */
const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

let A: Matrix;
let svd: SvdResult;
let svdReconMat: Matrix;
let sharedScale = 1;
let sigmaScale = 1;

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function sizeFromSlider(): number {
  const i = clamp(
    Math.round(Number(el.size.value) || 0),
    0,
    SIZE_STOPS.length - 1,
  );
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
  paint();
}

function paint(): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdReconMat, sharedScale);
  const err = frobeniusSq(sub(A, svdReconMat));
  el.svdErr.textContent = `‖A − U diag(σ) Vᵀ‖_F² = ${err.toExponential(3)}`;
}

el.regen.addEventListener("click", () => recompute(true));
el.size.addEventListener("input", () => recompute(true));

syncSliderLabels();
recompute(true);

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});
