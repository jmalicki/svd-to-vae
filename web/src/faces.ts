import "./style.css";
import { chapterNav } from "./chapterNav";
import { mountPage } from "./mountPage";
import pageHtml from "./pages/faces.html?raw";
import {
  FACE_SIZE,
  drawGray,
  getAppearanceCode,
  decodeAppearance,
  loadImmFaceBundle,
  mseGray,
  type FaceModel,
} from "./faceModel";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

mountPage(app, pageHtml, {
  nav: chapterNav({
    current: 4,
    prev: { href: "./truncate.html", label: "← Truncate" },
    next: { href: "./gradient.html", label: "Next →" },
  }),
});

const el = {
  loadStatus: app.querySelector<HTMLSpanElement>("#loadStatus")!,
  filmstrip: app.querySelector<HTMLDivElement>("#filmstrip")!,
  demoPanel: app.querySelector<HTMLElement>("#demoPanel")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLSpanElement>("#rankVal")!,
  kExplain: app.querySelector<HTMLParagraphElement>("#kExplain")!,
  reconCap: app.querySelector<HTMLSpanElement>("#reconCap")!,
  cvOrig: app.querySelector<HTMLCanvasElement>("#cvOrig")!,
  cvRecon: app.querySelector<HTMLCanvasElement>("#cvRecon")!,
  cvMean: app.querySelector<HTMLCanvasElement>("#cvMean")!,
  codeBars: app.querySelector<HTMLCanvasElement>("#codeBars")!,
};

let model: FaceModel | null = null;
let selected = 0;
/** Rank of the fitted model (= slider max); at most N−1 for N faces. */
let modelRank = 0;

function syncLabel(): void {
  el.rankVal.textContent = el.rank.value;
}

function explain(k: number, err: number): void {
  if (k === 0) {
    el.kExplain.textContent =
      "k = 0: you kept no numbers at all. Everyone rebuilds as the same average face — this person’s identity is gone.";
  } else if (modelRank > 0 && k >= modelRank) {
    el.kExplain.textContent = `k = ${k}: full rank for this stack (N−1 = ${modelRank}). Middle should match the left (error ${err.toFixed(4)}).`;
  } else if (k <= 3) {
    el.kExplain.textContent = `k = ${k}: only a few numbers fit through. You get a rough “someone,” not quite this person yet (error ${err.toFixed(4)} vs the left picture).`;
  } else if (k <= 12) {
    el.kExplain.textContent = `k = ${k}: more numbers get through, so lighting and features return. Closer to the left (error ${err.toFixed(4)}).`;
  } else if (k <= 40) {
    el.kExplain.textContent = `k = ${k}: most of this face fits, but this is still truncated — full rank is ${modelRank} (error ${err.toFixed(4)}).`;
  } else {
    el.kExplain.textContent = `k = ${k}: fine detail returning; only ${modelRank - k} directions left unused (error ${err.toFixed(4)}).`;
  }
  el.reconCap.textContent = k === 0 ? "average only" : `${k} number${k === 1 ? "" : "s"}`;
}

function drawCodeBars(code: Float64Array, k: number): void {
  const c = el.codeBars;
  const ctx = c.getContext("2d")!;
  const w = c.width;
  const h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const n = code.length;
  const gap = n > 80 ? 0 : 2;
  const barW = Math.max(1, (w - gap * (n + 1)) / n);
  let maxAbs = 1e-6;
  for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(code[i]!));
  const mid = h / 2;
  ctx.strokeStyle = "rgba(45,45,45,0.2)";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();
  for (let i = 0; i < n; i++) {
    const v = code[i]! / maxAbs;
    const bh = v * (h * 0.42);
    const x = gap + i * (barW + gap);
    const active = i < k;
    ctx.fillStyle = active ? "#0072b2" : "rgba(45,45,45,0.18)";
    if (bh >= 0) ctx.fillRect(x, mid - bh, barW, Math.max(bh, 1));
    else ctx.fillRect(x, mid, barW, Math.max(-bh, 1));
  }
}

function refresh(): void {
  if (!model) return;
  const k = Number(el.rank.value);
  const orig = model.examples[selected]!.appearance;
  const code = getAppearanceCode(model, selected);
  const recon = decodeAppearance(model, code, k);
  drawGray(el.cvOrig, orig, FACE_SIZE);
  drawGray(el.cvRecon, recon, FACE_SIZE);
  drawGray(el.cvMean, model.meanAppearance, FACE_SIZE);
  drawCodeBars(code, k);
  explain(k, mseGray(orig, recon));
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
      refresh();
    });
    el.filmstrip.appendChild(cell);
  });
}

el.rank.addEventListener("input", () => {
  syncLabel();
  refresh();
});

syncLabel();

function clearLoading(): void {
  el.filmstrip.classList.remove("is-loading");
  el.filmstrip.removeAttribute("aria-busy");
  el.demoPanel.classList.remove("is-loading");
  el.demoPanel.removeAttribute("aria-busy");
}

void (async () => {
  try {
    el.loadStatus.textContent = "Loading face pack…";
    const { model: loaded } = await loadImmFaceBundle(`${import.meta.env.BASE_URL}imm`, (msg) => {
      el.loadStatus.textContent = msg;
    });
    model = loaded;
    modelRank = model.appearanceU.cols;
    el.rank.max = String(modelRank);
    el.rank.disabled = false;
    clearLoading();
    el.loadStatus.textContent = `${model.examples.length} faces · click one, then scrub k`;
    paintStrip();
    drawGray(el.cvMean, model.meanAppearance, FACE_SIZE);
    refresh();
  } catch (err) {
    console.error(err);
    el.loadStatus.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
    el.filmstrip.classList.add("is-loading");
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
