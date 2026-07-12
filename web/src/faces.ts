import "./style.css";
import { chapterNav } from "./chapterNav";
import {
  FACE_SIZE,
  buildFaceModel,
  decodeAppearance,
  drawGray,
  getAppearanceCode,
  loadImmExamples,
  mseGray,
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
      current: 2,
      prev: { href: "./", label: "← Truncated SVD" },
      next: { href: "./gradient.html", label: "Next →" },
    })}
    <h1>Squeezing a face through a few numbers</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      On the
      <a href="./">previous page</a>,
      a
      <a href="https://en.wikipedia.org/wiki/Low-rank_approximation" target="_blank" rel="noopener noreferrer">truncated SVD</a>
      was a way to <strong>compress</strong> a matrix: keep only the top $k$ pieces, throw the rest away,
      and rebuild an approximation. Same $k$ idea here — but the “matrix” is a stack of face photos,
      and the approximation is a picture you can look at. A face is thousands of pixels; we ask how
      small $k$ can be before you stop recognizing <em>this</em> person.
    </p>
  </header>

  <section class="theory" aria-label="Plain story">
    <h2>From truncated SVD to a face</h2>
    <ol class="theory-steps">
      <li>
        <p>
          <strong>Compression means “keep $k$, discard the rest.”</strong>
          Truncated SVD finds the best way (in a least-squares sense) to store each example as
          $k$ numbers plus a shared decoder. Full data → thin middle of width $k$ → rebuilt copy.
          That thin middle is the bottleneck. Small $k$ = aggressive compression (blurrier /
          more generic). Large $k$ = milder compression (closer to the original).
        </p>
      </li>
      <li>
        <p>
          <strong>Line the faces up first</strong> (so compression isn’t wasted on “where is the eye?”).
          Raw photos have features in different places; compressing them as-is smears an eye onto a
          cheek. We mark the same spots on every face and gently stretch each photo so those spots
          land on the same places — the “warp.” Still a full-resolution picture, just lined up.
          Only the SVD step that follows is the actual compression.
        </p>
      </li>
      <li>
        <p>
          <strong>Rebuild</strong> means paint a picture from those $k$ numbers.
          Left = lined-up original. Middle = truncated-SVD reconstruction (rank $k$).
          Right = $k=0$: nothing kept, just the average face — maximum compression, no identity.
        </p>
      </li>
    </ol>
    <p class="theory-note">
      Drag $k$ below: you are literally changing the truncation rank. Names like Procrustes and
      Delaunay are in the <a href="#appendix">appendix</a>.
    </p>
  </section>

  <section class="panel" aria-label="Pick a face">
    <div class="panel-head">
      <h2>Pick a face</h2>
      <span id="loadStatus" class="muted">Loading…</span>
    </div>
    <p class="hint">
      Click someone below. Then move $k$: watch the middle picture go from “generic human” toward
      the person on the left.
    </p>
    <div id="filmstrip" class="filmstrip selectable" role="list"></div>
  </section>

  <section class="panel demo-hero" aria-label="Bottleneck demo">
    <h2>How many numbers does this face need?</h2>
    <div class="compare-row" id="compareRow">
      <figure>
        <canvas id="cvOrig" width="64" height="64"></canvas>
        <figcaption>Lined-up original<br /><span class="cap-sub">full picture</span></figcaption>
      </figure>
      <div class="compare-arrow" aria-hidden="true">
        <span class="arrow-label">keep only $k$<br />numbers</span>
        <span class="arrow-glyph">→</span>
      </div>
      <figure>
        <canvas id="cvRecon" width="64" height="64"></canvas>
        <figcaption>Rebuilt from $k$ numbers<br /><span class="cap-sub" id="reconCap">$k$ numbers</span></figcaption>
      </figure>
      <div class="compare-arrow" aria-hidden="true">
        <span class="arrow-label">keep nothing</span>
        <span class="arrow-glyph">→</span>
      </div>
      <figure>
        <canvas id="cvMean" width="64" height="64"></canvas>
        <figcaption>Average of everyone<br /><span class="cap-sub">$k = 0$</span></figcaption>
      </figure>
    </div>

    <div class="controls face-controls">
      <label>
        How many numbers ($k$)
        <input id="rank" type="range" min="0" max="30" step="1" value="8" disabled />
        <span id="rankVal" class="val">8</span>
      </label>
    </div>
    <p class="demo-callout" id="kExplain">
      Loading…
    </p>

    <h3 class="subhead">Those $k$ numbers, as bars</h3>
    <p class="hint">
      Each bar is one of the numbers in the short list for this face. Blue bars are the ones you are
      keeping. Gray bars are thrown away at this $k$ — that is the squeeze.
    </p>
    <canvas id="codeBars" class="code-bars" width="640" height="120" aria-label="Bottleneck coefficients"></canvas>
  </section>

  <section class="theory appendix" id="appendix" aria-label="Appendix">
    <h2>Appendix — names for the curious</h2>
    <ul class="appendix-list">
      <li>
        <strong>Warp / alignment.</strong>
        We use landmark
        <a href="https://en.wikipedia.org/wiki/Procrustes_analysis" target="_blank" rel="noopener noreferrer">Procrustes</a>
        alignment, then a piecewise-affine warp on a
        <a href="https://en.wikipedia.org/wiki/Delaunay_triangulation" target="_blank" rel="noopener noreferrer">Delaunay</a>
        mesh (<code>delaunator</code>) so every face shares one mean shape. The warp does
        <em>not</em> reduce dimension; it only changes coordinates.
      </li>
      <li>
        <strong>Bottleneck / rebuild.</strong>
        Same story as truncated SVD on the previous page: keep rank $k$, discard the rest, decode.
        Here the matrix columns are lined-up face images; the middle panel is $\\hat x$ from those
        $k$ coefficients (<code>ml-matrix</code> SVD).
      </li>
      <li>
        <strong>Data.</strong>
        <a href="http://www2.imm.dtu.dk/pubdb/pubs/3160-full.html" target="_blank" rel="noopener noreferrer">IMM Face Database</a>
        — full set of 240 landmarked faces; free for education/research; cite Stegmann et al. (FAME, IEEE TMI 2003).
      </li>
    </ul>
  </section>

  <footer class="chapter-footer">
    <a href="./">← Back: truncated SVD</a>
    <a href="./gradient.html">Next →</a>
  </footer>
`;

const el = {
  loadStatus: app.querySelector<HTMLSpanElement>("#loadStatus")!,
  filmstrip: app.querySelector<HTMLDivElement>("#filmstrip")!,
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
const MAX_K = 30;

function syncLabel(): void {
  el.rankVal.textContent = el.rank.value;
}

function explain(k: number, err: number): void {
  if (k === 0) {
    el.kExplain.textContent =
      "k = 0: you kept no numbers at all. Everyone rebuilds as the same average face — this person’s identity is gone.";
  } else if (k <= 3) {
    el.kExplain.textContent = `k = ${k}: only a few numbers fit through. You get a rough “someone,” not quite this person yet (error ${err.toFixed(4)} vs the left picture).`;
  } else if (k <= 12) {
    el.kExplain.textContent = `k = ${k}: more numbers get through, so lighting and features return. Closer to the left (error ${err.toFixed(4)}).`;
  } else {
    el.kExplain.textContent = `k = ${k}: most of this face fits. Extra numbers mostly add fine detail (error ${err.toFixed(4)}).`;
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
  const n = Math.min(code.length, MAX_K);
  const gap = 2;
  const barW = (w - gap * (n + 1)) / n;
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
    el.rank.max = String(fullK);
    el.rank.disabled = false;
    el.loadStatus.textContent = `${examples.length} faces · click one, then scrub k`;
    paintStrip();
    drawGray(el.cvMean, model.meanAppearance, FACE_SIZE);
    refresh();
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
