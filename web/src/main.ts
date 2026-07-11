import "./style.css";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import {
  SvdGradTrainer,
  probeGpu,
  type DeviceKind,
  type GradState,
} from "./svdGrad";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
} from "./matrix";
import { drawHeatmap, drawSigmaBars, drawLossChart, type LossPoint } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    <h1>SVD via gradient descent</h1>
    <p>
      Train factors with Adam and watch them approach truncated SVD.
      The reconstruction objective alone finds a best low-rank fit; orthonormality
      is what pins down the same $U$, $\\sigma$, and $V$ (up to signs).
    </p>
  </header>

  <section class="theory" aria-label="Loss construction">
    <h2>Building the loss</h2>
    <ol class="theory-steps">
      <li>
        <p>
          By the
          <a href="https://en.wikipedia.org/wiki/Low-rank_approximation#Eckart%E2%80%93Young%E2%80%93Mirsky_theorem" target="_blank" rel="noopener noreferrer">Eckart–Young–Mirsky theorem</a>,
          the best rank-$k$ approximation to $A$ in Frobenius norm is the truncated SVD
          $\\hat A_k = U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}$.
        </p>
        <div class="math">
          $$\\min_{\\mathrm{rank}(B)\\le k}\\;\\|A - B\\|_F
            = \\|A - U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}\\|_F$$
        </div>
      </li>
      <li>
        <p>
          So a natural training objective is reconstruction error with free factors
          $U\\in\\mathbb{R}^{n\\times k}$, $\\sigma\\in\\mathbb{R}^{k}$, $V\\in\\mathbb{R}^{n\\times k}$:
        </p>
        <div class="math">
          $$L_{\\mathrm{recon}} = \\bigl\\|A - U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}\\bigr\\|_F^{2}$$
        </div>
        <p class="theory-note">
          Orthogonality is <strong>not</strong> required for a best low-rank fit:
          many factorizations can realize the same $\\hat A_k$. Minimizing $L_{\\mathrm{recon}}$ alone
          targets the same reconstruction, but not necessarily the SVD bases.
        </p>
      </li>
      <li>
        <p>
          To recover the SVD factors themselves, add soft orthonormality penalties
          (columns of $U$ and $V$ should satisfy $U^{\\top}U \\approx I_k$ and $V^{\\top}V \\approx I_k$).
          Then $\\sigma$ plays the role of singular values:
        </p>
        <div class="math">
          $$L = \\underbrace{\\bigl\\|A - U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}\\bigr\\|_F^{2}}_{\\text{same low-rank approx}}
            + \\lambda\\Big(
              \\underbrace{\\|U^{\\top}U - I_k\\|_F^{2} + \\|V^{\\top}V - I_k\\|_F^{2}}_{\\text{same }U,V\\text{ (up to signs)}}
            \\Big)$$
        </div>
        <p class="theory-note">
          $\\lambda$ trades off fit vs.&nbsp;orthonormality. Without it you can match $\\hat A_k$;
          with it, gradient descent is steered toward the SVD’s $U$, $\\sigma$, and $V$.
        </p>
      </li>
    </ol>
  </section>

  <div class="controls">
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Size <em>n</em> <strong id="sizeVal">5</strong></span>
        <input id="size" type="range" min="3" max="16" step="1" value="5" />
      </label>
      <p class="help">Side length of random square matrix <code>A</code> (n×n). 4–8 is a good demo range.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Rank <em>k</em> <strong id="rankVal">3</strong></span>
        <input id="rank" type="range" min="1" max="5" step="1" value="3" />
      </label>
      <p class="help">How many singular components to keep (<code>1 ≤ k ≤ n</code>). Both columns use the same <em>k</em>; lower <em>k</em> is a coarser approximation.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">λ <strong id="lambdaVal">1.0</strong></span>
        <input id="lambda" type="range" min="0" max="5" step="0.1" value="1" />
      </label>
      <p class="help">Weight on soft orthogonality (<code>‖UᵀU−I‖</code>, <code>‖VᵀV−I‖</code>). Higher λ → more orthonormal factors; lower → reconstruction only.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Learning rate <strong id="lrVal">0.08</strong></span>
        <input id="lr" type="range" min="0.01" max="0.40" step="0.01" value="0.08" />
      </label>
      <p class="help">Adam step size. Too high oscillates; too low crawls. ~0.05–0.15 works well here.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Steps / frame <strong id="speedVal">2</strong></span>
        <input id="speed" type="range" min="1" max="40" step="1" value="2" />
      </label>
      <p class="help">Adam updates before each redraw. Use <strong>1</strong> to watch slowly; raise to converge faster.</p>
    </div>
    <div class="control-row">
      <label class="device">
        <span class="slider-label">Device</span>
        <select id="device">
          <option value="gpu">GPU (GPU.js / WebGL)</option>
          <option value="cpu">CPU</option>
        </select>
      </label>
      <p class="help">GPU.js over WebGL (not WebGPU). CPU is often faster for tiny matrices; falls back if WebGL fails.</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button id="play" type="button">Pause</button>
        <button id="reset" class="secondary" type="button">Reset</button>
      </div>
      <p class="help">Pause freezes training. Reset draws a new <code>A</code>, recomputes SVD, and re-inits GD.</p>
      <div class="status" id="status">step 0</div>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>Matrix A</h2>
      <canvas id="A" width="160" height="160"></canvas>
    </div>
    <div class="panel loss-wrap" style="flex:1;min-width:240px">
      <h2>Loss</h2>
      <canvas id="loss" width="520" height="140"></canvas>
      <div class="legend">
        <span><i style="background:#4C4C4C"></i>total</span>
        <span><i style="background:#0072B2"></i>recon</span>
        <span><i style="background:#E69F00"></i>ortho</span>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel" id="svdCol">
      <h2>Classical SVD (rank k)</h2>
      <div class="factor-row">
        <div class="factor"><span>Û</span><canvas id="svdU" width="100" height="100"></canvas></div>
        <div class="factor"><span>σ</span><canvas id="svdS" width="100" height="80"></canvas></div>
        <div class="factor"><span>V̂</span><canvas id="svdV" width="100" height="100"></canvas></div>
        <div class="factor"><span>Û diag(σ) V̂ᵀ</span><canvas id="svdRecon" width="120" height="120"></canvas></div>
      </div>
      <p class="note" id="svdErr"></p>
    </div>
    <div class="panel" id="gdCol">
      <h2>Gradient descent</h2>
      <div class="factor-row">
        <div class="factor"><span>U</span><canvas id="gdU" width="100" height="100"></canvas></div>
        <div class="factor"><span>σ</span><canvas id="gdS" width="100" height="80"></canvas></div>
        <div class="factor"><span>V</span><canvas id="gdV" width="100" height="100"></canvas></div>
        <div class="factor"><span>U diag(σ) Vᵀ</span><canvas id="gdRecon" width="120" height="120"></canvas></div>
      </div>
      <p class="note" id="gdErr"></p>
    </div>
  </div>

  <p class="note">
    Factors match SVD only up to sign flips and column order.
    GPU mode uses GPU.js over WebGL (not WebGPU). Small matrices often train faster on CPU.
  </p>
`;

const el = {
  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLElement>("#rankVal")!,
  device: app.querySelector<HTMLSelectElement>("#device")!,
  lambda: app.querySelector<HTMLInputElement>("#lambda")!,
  lambdaVal: app.querySelector<HTMLElement>("#lambdaVal")!,
  lr: app.querySelector<HTMLInputElement>("#lr")!,
  lrVal: app.querySelector<HTMLElement>("#lrVal")!,
  speed: app.querySelector<HTMLInputElement>("#speed")!,
  speedVal: app.querySelector<HTMLElement>("#speedVal")!,
  play: app.querySelector<HTMLButtonElement>("#play")!,
  reset: app.querySelector<HTMLButtonElement>("#reset")!,
  status: app.querySelector<HTMLDivElement>("#status")!,
  A: app.querySelector<HTMLCanvasElement>("#A")!,
  loss: app.querySelector<HTMLCanvasElement>("#loss")!,
  svdU: app.querySelector<HTMLCanvasElement>("#svdU")!,
  svdS: app.querySelector<HTMLCanvasElement>("#svdS")!,
  svdV: app.querySelector<HTMLCanvasElement>("#svdV")!,
  svdRecon: app.querySelector<HTMLCanvasElement>("#svdRecon")!,
  svdErr: app.querySelector<HTMLParagraphElement>("#svdErr")!,
  gdU: app.querySelector<HTMLCanvasElement>("#gdU")!,
  gdS: app.querySelector<HTMLCanvasElement>("#gdS")!,
  gdV: app.querySelector<HTMLCanvasElement>("#gdV")!,
  gdRecon: app.querySelector<HTMLCanvasElement>("#gdRecon")!,
  gdErr: app.querySelector<HTMLParagraphElement>("#gdErr")!,
};

let A: Matrix;
let svd: SvdResult;
let trainer = new SvdGradTrainer();
let gd: GradState;
let history: LossPoint[] = [];
let playing = true;
let raf = 0;
let sharedScale = 1;
let sigmaScale = 1;
let gpuOk = false;

function syncSliderLabels(): void {
  const n = clamp(Math.round(Number(el.size.value) || 5), 3, 16);
  el.size.value = String(n);
  el.rank.max = String(n);
  let k = clamp(Math.round(Number(el.rank.value) || 1), 1, n);
  el.rank.value = String(k);
  el.sizeVal.textContent = String(n);
  el.rankVal.textContent = String(k);
  el.lambdaVal.textContent = Number(el.lambda.value).toFixed(1);
  el.lrVal.textContent = Number(el.lr.value).toFixed(2);
  el.speedVal.textContent = String(Math.round(Number(el.speed.value)));
}

function readControls() {
  syncSliderLabels();
  const n = Number(el.size.value);
  const k = Number(el.rank.value);
  const device = el.device.value as DeviceKind;
  const lambda = Number(el.lambda.value);
  const lr = Number(el.lr.value);
  const steps = Number(el.speed.value);
  return { n, k, device, lambda, lr, steps };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function reset(newA: boolean): void {
  const { n, k, device, lambda, lr } = readControls();
  if (newA) {
    A = randomNormal(n, n, 1);
  } else if (!A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, k);
  const svdRecon = reconstruct(svd.U, svd.sigma, svd.V);
  sharedScale = Math.max(maxAbs(A), maxAbs(svdRecon), 1e-6);
  sigmaScale = Math.max(...svd.sigma, 1e-6);

  let useDevice: DeviceKind = device;
  if (useDevice === "gpu" && !gpuOk) {
    useDevice = "cpu";
    el.device.value = "cpu";
    el.status.textContent = "GPU unavailable — using CPU";
  }

  try {
    trainer.init(A, k, lambda, lr, useDevice);
  } catch (e) {
    console.warn("init failed on", useDevice, e);
    trainer.init(A, k, lambda, lr, "cpu");
    el.device.value = "cpu";
  }

  history = [];
  gd = trainer.snapshot(A);
  history.push({ ...gd.loss });
  paintSvd(svdRecon);
  paintGd();
  updateStatus();
}

function paintSvd(svdRecon: Matrix): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdRecon, sharedScale);
  const err = frobeniusSq(sub(A, svdRecon));
  el.svdErr.textContent = `‖A − Â_svd‖_F² = ${err.toExponential(3)}`;
}

function paintGd(): void {
  const recon = reconstruct(gd.U, gd.sigma, gd.V);
  drawHeatmap(el.gdU, gd.U);
  drawHeatmap(el.gdV, gd.V);
  drawSigmaBars(el.gdS, gd.sigma, sigmaScale);
  drawHeatmap(el.gdRecon, recon, sharedScale);
  drawLossChart(el.loss, history);
  const err = frobeniusSq(sub(A, recon));
  el.gdErr.textContent = `‖A − Â_gd‖_F² = ${err.toExponential(3)} · recon ${gd.loss.recon.toExponential(2)} · ortho ${gd.loss.ortho.toExponential(2)}`;
}

function updateStatus(): void {
  const { steps } = readControls();
  el.status.textContent = `step ${gd.step} · ${gd.device} · ${steps}/frame · L=${gd.loss.total.toExponential(3)}`;
}

function frame(): void {
  if (playing) {
    const { steps, lambda, lr } = readControls();
    trainer.setHyperparams(lambda, lr);
    try {
      for (let i = 0; i < steps; i++) {
        gd = trainer.stepOnce();
        history.push({ ...gd.loss });
        if (history.length > 600) history.shift();
      }
      paintGd();
      updateStatus();
    } catch (e) {
      console.error(e);
      playing = false;
      el.play.textContent = "Play";
      el.status.textContent = `error — paused (${String(e)})`;
    }
  }
  raf = requestAnimationFrame(frame);
}

el.play.addEventListener("click", () => {
  playing = !playing;
  el.play.textContent = playing ? "Pause" : "Play";
});

el.reset.addEventListener("click", () => reset(true));

el.size.addEventListener("input", () => {
  syncSliderLabels();
  reset(true);
});
el.rank.addEventListener("input", () => {
  syncSliderLabels();
  reset(false);
});
el.device.addEventListener("change", () => reset(false));
el.lambda.addEventListener("input", () => {
  syncSliderLabels();
  const { lambda, lr } = readControls();
  trainer.setHyperparams(lambda, lr);
});
el.lr.addEventListener("input", () => {
  syncSliderLabels();
  const { lambda, lr } = readControls();
  trainer.setHyperparams(lambda, lr);
});
el.speed.addEventListener("input", () => syncSliderLabels());

gpuOk = probeGpu();
if (!gpuOk) {
  el.device.value = "cpu";
}
syncSliderLabels();
reset(true);
raf = requestAnimationFrame(frame);

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  // MathJax may still be loading; retry once it arrives
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});

void raf;
