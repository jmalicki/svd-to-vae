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
    prev: { href: "./svd.html", label: "← Factors" },
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
