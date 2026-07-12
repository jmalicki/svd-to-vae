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
      current: 3,
      prev: { href: "./svd.html", label: "← Factors" },
      next: { href: "./faces.html", label: "Next →" },
    })}
    <h1>Keeping only some of the stretches</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      Last page built $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$ from orthogonal maps and stretches.
      Once you have many $\\sigma$'s, you can keep only the largest $k$ and rebuild an approximation
      $\\hat A_k = U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}$ — that is
      <a href="https://en.wikipedia.org/wiki/Low-rank_approximation" target="_blank" rel="noopener noreferrer">truncated SVD</a>.
    </p>
  </header>

  <section class="theory" aria-label="Truncated SVD">
    <h2>What you are looking at</h2>
    <ol class="theory-steps">
      <li>
        <p>
          Start from the full factorization $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$.
          The singular values are ordered $\\sigma_1\\ge\\sigma_2\\ge\\cdots\\ge 0$.
        </p>
      </li>
      <li>
        <p>
          <strong>Truncate</strong> means keep rank $k$: take the first $k$ columns of $U$ and $V$
          and the first $k$ singular values, discard the rest, and multiply back out.
        </p>
        <div class="math">
          $$\\hat A_k = U_k\\,\\mathrm{diag}(\\sigma_{1:k})\\,V_k^{\\top}$$
        </div>
        <p>
          Lower $k$ throws away more detail; higher $k$ is closer to $A$.
          The live error $\\|A - \\hat A_k\\|_F^{2}$ measures what was lost.
        </p>
      </li>
      <li>
        <p>
          By the
          <a href="https://en.wikipedia.org/wiki/Low-rank_approximation#Eckart%E2%80%93Young%E2%80%93Mirsky_theorem" target="_blank" rel="noopener noreferrer">Eckart–Young–Mirsky theorem</a>,
          this truncated reconstruction is the best possible rank-$k$ approximation in Frobenius norm.
          See the <a href="#appendix">appendix</a> for orthonormal columns, singular values, and the norm.
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
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Rank <em>k</em> <strong id="rankVal">3</strong></span>
        <input id="rank" type="range" min="1" max="5" step="1" value="3" />
      </label>
      <p class="help">How many singular components to keep (<code>1 ≤ k ≤ n</code>). Lower <em>k</em> is a coarser approximation.</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button id="regen" type="button">Regenerate A</button>
      </div>
      <p class="help">Draw a new random Gaussian matrix and recompute the truncated SVD.</p>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>Matrix A</h2>
      <canvas id="A" width="160" height="160"></canvas>
    </div>
  </div>

  <div class="panel" id="svdCol">
    <h2>Truncated SVD (rank k)</h2>
    <div class="factor-row">
      <div class="factor"><span>Û</span><canvas id="svdU" width="100" height="100"></canvas></div>
      <div class="factor"><span>σ</span><canvas id="svdS" width="100" height="80"></canvas></div>
      <div class="factor"><span>V̂</span><canvas id="svdV" width="100" height="100"></canvas></div>
      <div class="factor"><span>Û diag(σ) V̂ᵀ</span><canvas id="svdRecon" width="120" height="120"></canvas></div>
    </div>
    <p class="note" id="svdErr"></p>
  </div>

  <section class="appendix" id="appendix" aria-label="Appendix">
    <h2>Appendix: vocabulary</h2>

    <h3>Orthonormal columns</h3>
    <p>
      Columns of $U$ and $V$ have unit length and are mutually perpendicular:
      $U^{\\top}U = I_k$ and $V^{\\top}V = I_k$.
      Geometrically, $U$ and $V$ are rigid rotations (or reflections) of the coordinate axes —
      they do not stretch space, only reorient it.
    </p>

    <h3>Singular values</h3>
    <p>
      The numbers $\\sigma_1\\ge\\sigma_2\\ge\\cdots$ measure how much $A$ stretches space along each
      orthogonal direction. Large $\\sigma_j$ means that direction carries a lot of energy;
      tiny $\\sigma_j$ means it is nearly redundant.
      Truncation drops the smallest ones first.
    </p>

    <h3>Frobenius norm</h3>
    <p>
      The
      <a href="https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm" target="_blank" rel="noopener noreferrer">Frobenius norm</a>
      treats a matrix like a long vector of entries:
    </p>
    <div class="math">
      $$\\|M\\|_F = \\sqrt{\\sum_{i,j} M_{ij}^{2}},\\qquad
        \\|M\\|_F^{2} = \\sum_{i,j} M_{ij}^{2}.$$
    </div>
    <p>
      Squared Frobenius error is the sum of squared entrywise differences — the same notion as
      mean squared error when every entry counts equally.
    </p>

    <h3>Eckart–Young</h3>
    <p>
      Among all rank-$k$ matrices $B$, the truncated SVD reconstruction $\\hat A_k$ uniquely minimizes
      $\\|A - B\\|_F$ (and therefore $\\|A - B\\|_F^{2}$).
      No other rank-$k$ factorization can beat it in least-squares reconstruction error.
    </p>
    <div class="math">
      $$\\min_{\\mathrm{rank}(B)\\le k}\\;\\|A - B\\|_F
        = \\|A - U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}\\|_F$$
    </div>
  </section>

  <section class="conclusion" id="conclusion" aria-label="Summary">
    <h2>In short</h2>
    <p>
      Truncated SVD keeps the largest $k$ singular values and drops the rest.
      The reconstruction $\\hat A_k$ is the best rank-$k$ approximation in Frobenius norm.
    </p>
    <p class="next-chapter">
      <a href="./faces.html">Same idea, but the matrix is a stack of faces →</a>
    </p>
  </section>
`;

const el = {
  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLElement>("#rankVal")!,
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
  const n = sizeFromSlider();
  el.rank.max = String(n);
  const k = clamp(Math.round(Number(el.rank.value) || 1), 1, n);
  el.rank.value = String(k);
  el.sizeVal.textContent = String(n);
  el.rankVal.textContent = String(k);
}

function readControls(): { n: number; k: number } {
  syncSliderLabels();
  return { n: sizeFromSlider(), k: Number(el.rank.value) };
}

function recompute(newA: boolean): void {
  const { n, k } = readControls();
  if (newA || !A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, k);
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
  el.svdErr.textContent = `‖A − Â_k‖_F² = ${err.toExponential(3)}`;
}

el.regen.addEventListener("click", () => recompute(true));

el.size.addEventListener("input", () => recompute(true));

el.rank.addEventListener("input", () => recompute(false));

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
