import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import pageHtml from "./pages/noise.html?raw";
import {
  FACE_SIZE,
  decodeAppearance,
  drawGray,
  getAppearanceCode,
  loadImmFaceBundle,
  sampleNoisyAppearance,
  samplePixelFoil,
  type FaceModel,
  type PixelFoilModel,
} from "./faceModel";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

mountPage(app, pageHtml, {
  nav: chapterNav({
    current: 6,
    prev: { href: "./gradient.html", label: "← Gradient" },
  }),
});

const el = {
  loadStatus: app.querySelector<HTMLSpanElement>("#loadStatus")!,
  filmstrip: app.querySelector<HTMLDivElement>("#filmstrip")!,
  noiseRow: app.querySelector<HTMLDivElement>("#noiseRow")!,
  gallery: app.querySelector<HTMLDivElement>("#gallery")!,
  foilGallery: app.querySelector<HTMLDivElement>("#foilGallery")!,
  cvExact: app.querySelector<HTMLCanvasElement>("#cvExact")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  tau: app.querySelector<HTMLInputElement>("#tau")!,
  rankVal: app.querySelector<HTMLSpanElement>("#rankVal")!,
  tauVal: app.querySelector<HTMLSpanElement>("#tauVal")!,
  tauExplain: app.querySelector<HTMLParagraphElement>("#tauExplain")!,
  resample: app.querySelector<HTMLButtonElement>("#resample")!,
  resampleGallery: app.querySelector<HTMLButtonElement>("#resampleGallery")!,
};

const VARIANT_N = 5;
const GALLERY_N = 12;

let model: FaceModel | null = null;
let foil: PixelFoilModel | null = null;
let selected = 0;

function randn(): number {
  const u = Math.max(1e-12, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function noisyCode(modelIn: FaceModel, idx: number, k: number, tau: number): Float64Array {
  const base = getAppearanceCode(modelIn, idx);
  const kk = Math.min(k, base.length);
  const out = new Float64Array(kk);
  for (let ell = 0; ell < kk; ell++) {
    const scale = modelIn.appearanceSigma[ell] ?? 1;
    out[ell] = base[ell]! + tau * scale * randn();
  }
  return out;
}

function syncLabels(): void {
  el.rankVal.textContent = el.rank.value;
  el.tauVal.textContent = Number(el.tau.value).toFixed(2);
}

function explainTau(tau: number): void {
  if (tau < 0.05) {
    el.tauExplain.textContent =
      "τ ≈ 0: noise is off. Every “variant” matches the exact bottleneck reconstruction of this face.";
  } else if (tau < 0.35) {
    el.tauExplain.textContent = `τ = ${tau.toFixed(2)}: small jiggle — same person-ish, slightly different lighting or expression.`;
  } else if (tau < 0.9) {
    el.tauExplain.textContent = `τ = ${tau.toFixed(2)}: stronger wander in code space — still face-like, clearly not a pixel blur of the original.`;
  } else {
    el.tauExplain.textContent = `τ = ${tau.toFixed(2)}: far from $z$ — odd hybrids. Turn τ down to stay near the real face.`;
  }
}

function paintStrip(): void {
  if (!model) return;
  el.filmstrip.replaceChildren();
  model.examples.forEach((ex, idx) => {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "film-cell select-cell" + (idx === selected ? " selected" : "");
    cell.setAttribute("role", "listitem");
    cell.title = ex.id;
    const c = document.createElement("canvas");
    drawGray(c, ex.appearance, FACE_SIZE);
    cell.appendChild(c);
    cell.addEventListener("click", () => {
      selected = idx;
      paintStrip();
      refreshFocus();
    });
    el.filmstrip.appendChild(cell);
  });
}

function fillHost(host: HTMLDivElement, n: number, sample: () => Float64Array): void {
  host.replaceChildren();
  for (let i = 0; i < n; i++) {
    const cell = document.createElement("div");
    cell.className = "film-cell";
    cell.setAttribute("role", "listitem");
    const c = document.createElement("canvas");
    drawGray(c, sample(), FACE_SIZE);
    cell.appendChild(c);
    host.appendChild(cell);
  }
}

function refreshFocus(): void {
  if (!model) return;
  const k = Number(el.rank.value);
  const tau = Number(el.tau.value);
  const exact = decodeAppearance(model, getAppearanceCode(model, selected), k);
  drawGray(el.cvExact, exact, FACE_SIZE);
  fillHost(el.noiseRow, VARIANT_N, () =>
    decodeAppearance(model!, noisyCode(model!, selected, k, tau), k),
  );
  explainTau(tau);
}

function refreshGallery(): void {
  if (!model || !foil) return;
  const k = Number(el.rank.value);
  const tau = Number(el.tau.value);
  fillHost(el.gallery, GALLERY_N, () => sampleNoisyAppearance(model!, k, tau));
  fillHost(el.foilGallery, GALLERY_N, () => samplePixelFoil(foil!, k, tau));
}

function onControls(): void {
  syncLabels();
  refreshFocus();
  refreshGallery();
}

el.rank.addEventListener("input", onControls);
el.tau.addEventListener("input", onControls);
el.resample.addEventListener("click", () => refreshFocus());
el.resampleGallery.addEventListener("click", () => refreshGallery());

syncLabels();

void (async () => {
  try {
    el.loadStatus.textContent = "Loading face pack…";
    const loaded = await loadImmFaceBundle(`${import.meta.env.BASE_URL}imm`, (msg) => {
      el.loadStatus.textContent = msg;
    });
    model = loaded.model;
    foil = loaded.foil;
    const fullK = model.appearanceU.cols;
    el.rank.max = String(fullK);
    el.rank.disabled = false;
    el.tau.disabled = false;
    el.resample.disabled = false;
    el.resampleGallery.disabled = false;
    el.loadStatus.textContent = `${model.examples.length} faces ready`;
    paintStrip();
    onControls();
  } catch (err) {
    console.error(err);
    el.loadStatus.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
  }
})();

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});
