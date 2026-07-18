import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import pageHtml from "./pages/svdMath.html?raw";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
  takeColumns,
} from "./matrix";
import { drawHeatmap, drawSigmaBars } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

mountPage(app, pageHtml, {
  nav: chapterNav({
    current: 3,
    prev: { href: "./reflections.html", label: "← Reflections" },
    next: { href: "./faces.html", label: "Next →" },
  }),
});

const el = {
  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLElement>("#rankVal")!,
  regen: app.querySelector<HTMLButtonElement>("#regen")!,
  A: app.querySelector<HTMLCanvasElement>("#A")!,
  fullU: app.querySelector<HTMLCanvasElement>("#fullU")!,
  fullS: app.querySelector<HTMLCanvasElement>("#fullS")!,
  fullV: app.querySelector<HTMLCanvasElement>("#fullV")!,
  fullRecon: app.querySelector<HTMLCanvasElement>("#fullRecon")!,
  fullErr: app.querySelector<HTMLParagraphElement>("#fullErr")!,
  svdU: app.querySelector<HTMLCanvasElement>("#svdU")!,
  svdS: app.querySelector<HTMLCanvasElement>("#svdS")!,
  svdV: app.querySelector<HTMLCanvasElement>("#svdV")!,
  svdRecon: app.querySelector<HTMLCanvasElement>("#svdRecon")!,
  svdResid: app.querySelector<HTMLCanvasElement>("#svdResid")!,
  svdErr: app.querySelector<HTMLParagraphElement>("#svdErr")!,
};

/** Log-spaced matrix sizes for the n slider (index → n). */
const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

let A: Matrix;
let fullSvd: SvdResult;
let fullReconMat: Matrix;
let truncReconMat: Matrix;
let residMat: Matrix;
let sharedScale = 1;
let sigmaScale = 1;

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function sizeFromSlider(): number {
  const i = clamp(Math.round(Number(el.size.value) || 0), 0, SIZE_STOPS.length - 1);
  el.size.value = String(i);
  return SIZE_STOPS[i]!;
}

function syncSliderLabels(): { n: number; k: number } {
  const n = sizeFromSlider();
  el.rank.max = String(n);
  const k = clamp(Math.round(Number(el.rank.value) || 1), 1, n);
  el.rank.value = String(k);
  el.sizeVal.textContent = String(n);
  el.rankVal.textContent = String(k);
  return { n, k };
}

function truncateFactors(full: SvdResult, k: number): SvdResult {
  return {
    U: takeColumns(full.U, k),
    sigma: full.sigma.slice(0, k),
    V: takeColumns(full.V, k),
  };
}

function recompute(newA: boolean): void {
  const { n, k } = syncSliderLabels();
  if (newA || !A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  fullSvd = classicalSvd(A, n);
  fullReconMat = reconstruct(fullSvd.U, fullSvd.sigma, fullSvd.V);
  const trunc = truncateFactors(fullSvd, k);
  truncReconMat = reconstruct(trunc.U, trunc.sigma, trunc.V);
  residMat = sub(A, truncReconMat);
  sharedScale = Math.max(maxAbs(A), maxAbs(fullReconMat), maxAbs(truncReconMat), 1e-6);
  sigmaScale = Math.max(...fullSvd.sigma, 1e-6);
  paint();
}

function paint(): void {
  const k = Number(el.rank.value);
  const trunc = truncateFactors(fullSvd, k);

  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.fullU, fullSvd.U);
  drawHeatmap(el.fullV, fullSvd.V);
  drawSigmaBars(el.fullS, fullSvd.sigma, sigmaScale);
  drawHeatmap(el.fullRecon, fullReconMat, sharedScale);
  el.fullErr.textContent = `‖A − product‖_F² = ${frobeniusSq(sub(A, fullReconMat)).toExponential(3)}`;

  drawHeatmap(el.svdU, trunc.U);
  drawHeatmap(el.svdV, trunc.V);
  drawSigmaBars(el.svdS, trunc.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, truncReconMat, sharedScale);
  const residScale = Math.max(maxAbs(residMat), sharedScale * 0.15, 1e-6);
  drawHeatmap(el.svdResid, residMat, residScale);

  const err = frobeniusSq(residMat);
  const tail = fullSvd.sigma.slice(k).reduce((s, x) => s + x * x, 0);
  el.svdErr.textContent =
    `‖A − Â_k‖_F² = ${err.toExponential(3)}  ·  ∑_{j>k} σⱼ² = ${tail.toExponential(3)}`;
}

el.regen.addEventListener("click", () => recompute(true));
el.size.addEventListener("input", () => recompute(true));
el.rank.addEventListener("input", () => {
  const { k } = syncSliderLabels();
  const trunc = truncateFactors(fullSvd, k);
  truncReconMat = reconstruct(trunc.U, trunc.sigma, trunc.V);
  residMat = sub(A, truncReconMat);
  paint();
});

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
