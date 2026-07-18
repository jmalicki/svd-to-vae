import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import { SITE, TOUR_URL } from "./site";
import pageHtml from "./pages/gradient.html?raw";
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

// On the standalone svd-grad site this page is the whole site: swap the
// chapter strip for a pointer to the tour and send chapter links there.
const nav =
  SITE === "grad"
    ? `<p class="chapter-nav">Standalone demo ·
        <a href="${TOUR_URL}gradient.html">Part of the SVD-to-VAE tour →</a></p>`
    : chapterNav({
        current: 5,
        prev: { href: "./faces.html", label: "← Faces" },
        next: { href: "./noise.html", label: "Next →" },
      });

const html = SITE === "grad" ? pageHtml.replaceAll('href="./', `href="${TOUR_URL}`) : pageHtml;

mountPage(app, html, {
  nav,
  baseUrl: import.meta.env.BASE_URL,
});

const el = {
  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLElement>("#rankVal")!,
  device: app.querySelector<HTMLSelectElement>("#device")!,
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
  sigmaTable: app.querySelector<HTMLTableElement>("#sigmaTable")!,
  fpNote: app.querySelector<HTMLAnchorElement>("#fpNote")!,
  fpFloorVal: app.querySelector<HTMLElement>("#fpFloorVal")!,
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
let svdFloor = 0;
let svdReconMat: Matrix;
let gpuOk = false;

/** Log-spaced matrix sizes for the n slider (index → n). */
const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64];

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
  let k = clamp(Math.round(Number(el.rank.value) || 1), 1, n);
  el.rank.value = String(k);
  el.sizeVal.textContent = String(n);
  el.rankVal.textContent = String(k);
  el.lrVal.textContent = Number(el.lr.value).toFixed(3).replace(/0$/, "");
  el.speedVal.textContent = String(Math.round(Number(el.speed.value)));
}

function readControls() {
  syncSliderLabels();
  const n = sizeFromSlider();
  const k = Number(el.rank.value);
  const device = el.device.value as DeviceKind;
  const lr = Number(el.lr.value);
  const steps = Number(el.speed.value);
  return { n, k, device, lr, steps };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function reset(newA: boolean): void {
  const { n, k, device, lr } = readControls();
  if (newA) {
    A = randomNormal(n, n, 1);
  } else if (!A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, k);
  svdReconMat = reconstruct(svd.U, svd.sigma, svd.V);
  sharedScale = Math.max(maxAbs(A), maxAbs(svdReconMat), 1e-6);
  sigmaScale = Math.max(...svd.sigma, 1e-6);
  svdFloor = frobeniusSq(sub(A, svdReconMat));
  const ulp = Number.EPSILON * Math.max(maxAbs(A), Number.EPSILON);
  const fpFloor = A.rows * A.cols * ulp * ulp;
  const fpLabel = fpFloor.toExponential(1);
  el.fpNote.textContent = fpLabel;
  el.fpFloorVal.textContent = fpLabel;

  let useDevice: DeviceKind = device;
  if (useDevice === "gpu" && !gpuOk) {
    useDevice = "cpu";
    el.device.value = "cpu";
    el.status.textContent = "GPU unavailable — using CPU";
  }

  try {
    trainer.init(A, k, lr, useDevice);
  } catch (e) {
    console.warn("init failed on", useDevice, e);
    trainer.init(A, k, lr, "cpu");
    el.device.value = "cpu";
  }

  history = [];
  gd = trainer.snapshot(A);
  history.push(lossSample(gd));
  paintSvd();
  paintGd();
  updateStatus();
}

function lossSample(state: GradState): LossPoint {
  const gdRecon = reconstruct(state.U, state.sigma, state.V);
  return {
    step: state.step,
    recon: state.loss.recon,
    vsSvd: frobeniusSq(sub(svdReconMat, gdRecon)),
  };
}

function paintSvd(): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdReconMat, sharedScale);
  el.svdErr.textContent = `‖A − Â_svd‖_F² = ${svdFloor.toExponential(3)}`;
  paintSigmaTable();
}

function paintGd(): void {
  const recon = reconstruct(gd.U, gd.sigma, gd.V);
  drawHeatmap(el.gdU, gd.U);
  drawHeatmap(el.gdV, gd.V);
  drawSigmaBars(el.gdS, gd.sigma, sigmaScale);
  drawHeatmap(el.gdRecon, recon, sharedScale);
  drawLossChart(el.loss, history, svdFloor);
  const err = frobeniusSq(sub(A, recon));
  const vs = frobeniusSq(sub(svdReconMat, recon));
  el.gdErr.textContent = `‖A − Â_gd‖_F² = ${err.toExponential(3)}  ·  ‖Â_svd − Â_gd‖_F² = ${vs.toExponential(3)}  ·  SVD ${svdFloor.toExponential(3)}`;
  paintSigmaTable();
}

function fmtSigma(x: number): string {
  if (!Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e4 || a < 1e-4)) return x.toExponential(8);
  return x.toFixed(8);
}

/** Rows = SVD / GD; columns = σ₁…σₖ. */
function paintSigmaTable(): void {
  const k = svd.sigma.length;
  const thead = el.sigmaTable.querySelector("thead")!;
  const tbody = el.sigmaTable.querySelector("tbody")!;

  const headCells = ["", ...Array.from({ length: k }, (_, j) => String(j + 1))];
  thead.innerHTML = `<tr>${headCells.map((c) => `<th scope="col">${c}</th>`).join("")}</tr>`;

  const svdCells = svd.sigma.map((s) => `<td>${fmtSigma(s)}</td>`).join("");
  const gdCells = Array.from({ length: k }, (_, j) => {
    const s = gd.sigma[j];
    return `<td>${s === undefined ? "—" : fmtSigma(s)}</td>`;
  }).join("");

  tbody.innerHTML = `
    <tr><th scope="row">SVD</th>${svdCells}</tr>
    <tr><th scope="row">GD</th>${gdCells}</tr>
  `;
}

function updateStatus(): void {
  const { steps } = readControls();
  const eta = gd.lr.toExponential(2);
  el.status.textContent = `step ${gd.step} · ${gd.device} · ${steps}/frame · η=${eta} · L=${gd.loss.recon.toExponential(3)}`;
}

function frame(): void {
  if (playing) {
    const { steps, lr } = readControls();
    trainer.setLr(lr);
    try {
      for (let i = 0; i < steps; i++) {
        gd = trainer.stepOnce();
        history.push(lossSample(gd));
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
el.lr.addEventListener("input", () => {
  syncSliderLabels();
  const { lr } = readControls();
  trainer.setLr(lr);
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
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});

void raf;
