import "./style.css";
import { chapterNav } from "./chapterNav";
import {
  FACE_SIZE,
  buildFaceModel,
  buildPixelFoilModel,
  decodeAppearance,
  drawGray,
  getAppearanceCode,
  loadImmExamples,
  sampleNoisyAppearance,
  samplePixelFoil,
  type FaceModel,
  type ImmManifest,
} from "./faceModel";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    ${chapterNav({
      current: 5,
      prev: { href: "./gradient.html", label: "← Gradient" },
    })}
    <h1>Noise in the bottleneck</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      On the
      <a href="./faces.html">faces page</a>,
      each picture was squeezed through $k$ numbers $z$ and decoded again — that was compression.
      Now keep those $k$ numbers, but <strong>jiggle them</strong>:
      $\\tilde z = z + \\tau\\varepsilon$.
      Decoding $\\tilde z$ invents a nearby face. $\\tau=0$ is the exact reconstruction; bigger $\\tau$ wanders farther.
    </p>
  </header>

  <section class="panel" aria-label="Pick a face">
    <div class="panel-head">
      <h2>Start from one real face</h2>
      <span id="loadStatus" class="muted">Loading…</span>
    </div>
    <p class="hint">Click a face. Its bottleneck code $z$ is the center of the noise cloud.</p>
    <div id="filmstrip" class="filmstrip selectable" role="list"></div>
  </section>

  <section class="panel demo-hero" aria-label="Noise demo">
    <h2>What the noise does</h2>
    <div class="compare-row noise-row">
      <figure>
        <canvas id="cvExact" width="64" height="64"></canvas>
        <figcaption>Exact decode<br /><span class="cap-sub">$\\tau = 0$ (same $z$)</span></figcaption>
      </figure>
      <div class="compare-arrow" aria-hidden="true">
        <span class="arrow-label">add $\\tau\\varepsilon$</span>
        <span class="arrow-glyph">→</span>
      </div>
      <div id="noiseRow" class="noise-variants" role="list"></div>
    </div>

    <div class="controls face-controls">
      <label>
        Bottleneck $k$ (same idea as faces)
        <input id="rank" type="range" min="2" max="30" step="1" value="12" disabled />
        <span id="rankVal" class="val">12</span>
      </label>
      <label>
        Noise amount $\\tau$
        <input id="tau" type="range" min="0" max="1.5" step="0.05" value="0.4" disabled />
        <span id="tauVal" class="val">0.40</span>
      </label>
    </div>
    <p class="demo-callout" id="tauExplain">Loading…</p>
    <div class="panel-head">
      <h3 class="subhead" style="margin:0">Resample the same $z$ with fresh $\\varepsilon$</h3>
      <button type="button" id="resample" class="primary" disabled>Resample noise</button>
    </div>
  </section>

  <section class="panel" aria-label="Random gallery">
    <div class="panel-head">
      <h2>Random nearby faces</h2>
      <button type="button" id="resampleGallery" class="primary" disabled>Resample gallery</button>
    </div>
    <p class="hint">
      Each tile: pick a random training $z$, add noise, decode. Same warp-aligned SVD as the
      <a href="./faces.html">faces page</a>.
    </p>
    <div id="gallery" class="face-gallery" role="list"></div>
  </section>

  <section class="panel" aria-label="Pixel foil">
    <h2>Foil: same noise, no warp</h2>
    <p class="hint">
      Identical $k$ and $\\tau$ on loose bbox crops (no mean-shape warp). Draws go foggy — eigenfaces without correspondence.
    </p>
    <div id="foilGallery" class="face-gallery foil" role="list"></div>
  </section>

  <footer class="chapter-footer">
    <a href="./gradient.html">← Back: gradient</a>
    <a href="./faces.html">Faces demo</a>
    <a href="./">Home: truncated SVD</a>
  </footer>
`;

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
const MAX_K = 30;

let model: FaceModel | null = null;
let foil: ReturnType<typeof buildPixelFoilModel> | null = null;
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
    const manifest = (await fetch(`${import.meta.env.BASE_URL}imm/manifest.json`).then((r) =>
      r.json(),
    )) as ImmManifest;
    el.loadStatus.textContent = `Loading ${manifest.files.length} faces…`;
    const { examples } = await loadImmExamples(
      `${import.meta.env.BASE_URL}imm`,
      manifest.files,
      FACE_SIZE,
      (msg) => {
        el.loadStatus.textContent = msg;
      },
    );
    const fullK = Math.min(MAX_K, examples.length - 1);
    model = buildFaceModel(examples, fullK);
    foil = buildPixelFoilModel(examples, fullK);
    el.rank.max = String(fullK);
    el.rank.disabled = false;
    el.tau.disabled = false;
    el.resample.disabled = false;
    el.resampleGallery.disabled = false;
    el.loadStatus.textContent = `${examples.length} faces ready`;
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
