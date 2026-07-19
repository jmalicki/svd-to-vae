/**
 * Build-time ablation: global-RMS SGD, 2×2 over {v† floor on/off} × {Armijo on/off}.
 * Plots ‖Â_svd − Â_gd‖_F² (the amber curve in the live demo) for all four runs.
 * Writes public/ringing-floor.svg and exits nonzero if the neither-fix run does not
 * ring or the floored+Armijo run fails to stay near the Eckart–Young gap.
 *
 * These four cells use the raw Euclidean gradient (no tangent projection) so the
 * floor/Armijo failure modes stay visible; the projection's own 2×2×2 cube is
 * measured by scripts/gen-freeze-plot.mjs. Both scripts share the training loop
 * in scripts/ablation-lib.mjs.
 *
 * Run: node scripts/gen-ringing-plot.mjs
 */
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  mulberry32,
  randn,
  thinQ,
  train,
  truncSvd,
  renderSvg,
} from "./ablation-lib.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, "../public/ringing-floor.svg");

const N = 6;
const K = 3;
const STEPS = 900;
const BASE_LR = 0.05;
const SEED = 7;

/** Init used by this figure since its first version: U, V, then raw from one stream. */
function figureInit(seed) {
  const rand = mulberry32(seed);
  const U = thinQ(randn(rand, N, K, 0.3));
  const V = thinQ(randn(rand, N, K, 0.3));
  const raw = new Float64Array(K);
  for (let j = 0; j < K; j++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    raw[j] = 0.3 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return { U, raw, V };
}

/**
 * Ringing on ‖Â_svd − Â_gd‖_F²: after warm-up, many local peaks and large
 * peak-to-trough swing relative to the best (smallest) gap in the tail.
 */
function assertRings(series) {
  const warm = Math.floor(STEPS * 0.35);
  const tail = series.slice(warm);
  let minL = Infinity;
  let maxL = -Infinity;
  for (const L of tail) {
    minL = Math.min(minL, L);
    maxL = Math.max(maxL, L);
  }
  let peaks = 0;
  for (let i = 2; i < tail.length - 2; i++) {
    if (
      tail[i] > tail[i - 1] &&
      tail[i] > tail[i + 1] &&
      tail[i] > tail[i - 2] &&
      tail[i] > tail[i + 2]
    ) {
      peaks++;
    }
  }
  const swing = maxL - minL;
  const relSwing = swing / Math.max(minL, 1e-18);

  if (peaks < 8) {
    throw new Error(
      `expected unfloored ‖Â_svd−Â_gd‖² to ring (local peaks≥8 in tail), got peaks=${peaks}`,
    );
  }
  if (relSwing < 5) {
    throw new Error(
      `expected unfloored relative swing≥5×min, got ${relSwing.toFixed(3)} ` +
        `(swing=${swing.toExponential(3)}, min=${minL.toExponential(3)}, max=${maxL.toExponential(3)})`,
    );
  }
  return { peaks, swing, relSwing, minL, maxL };
}

function assertSettles(floored, unfloored) {
  const warm = Math.floor(STEPS * 0.35);
  const fTail = floored.slice(warm);
  const uTail = unfloored.slice(warm);
  let fMin = Infinity;
  let fMax = -Infinity;
  let uMin = Infinity;
  let uMax = -Infinity;
  for (const L of fTail) {
    fMin = Math.min(fMin, L);
    fMax = Math.max(fMax, L);
  }
  for (const L of uTail) {
    uMin = Math.min(uMin, L);
    uMax = Math.max(uMax, L);
  }
  const fSwing = fMax - fMin;
  const uSwing = uMax - uMin;
  const fRel = fSwing / Math.max(fMin, 1e-18);
  const finalL = floored[floored.length - 1];

  // Near EY, absolute levels are ~0 so relative swing is meaningless — require
  // a flat absolute tail (and final gap near 0). Farther out, also cap rel swing.
  if (fMax > 1e-6 && fRel > 0.1) {
    throw new Error(
      `floored ‖Â_svd−Â_gd‖² should settle (rel swing≤0.1 in tail); got ${fRel.toFixed(3)}`,
    );
  }
  if (!(fSwing < 0.05 * uSwing)) {
    throw new Error(
      `floored swing should be ≪ unfloored swing; ` +
        `fSwing=${fSwing.toExponential(3)} uSwing=${uSwing.toExponential(3)}`,
    );
  }
  if (finalL > 1e-4) {
    throw new Error(
      `floored+Armijo should keep ‖Â_svd−Â_gd‖² near 0; final=${finalL.toExponential(3)}`,
    );
  }
  return { swing: fSwing, relSwing: fRel, finalL, uMax, uSwing };
}

function main() {
  const rand = mulberry32(SEED);
  const A = randn(rand, N, N, 1);
  const Asvd = truncSvd(A, K);
  const init = figureInit(SEED + 1);
  const cfg = { k: K, steps: STEPS, baseLr: BASE_LR, project: false };

  const both = train(A, Asvd, { ...cfg, floor: true, armijo: true }, init);
  const floorOnly = train(A, Asvd, { ...cfg, floor: true, armijo: false }, init);
  const armijoOnly = train(A, Asvd, { ...cfg, floor: false, armijo: true }, init);
  const neither = train(A, Asvd, { ...cfg, floor: false, armijo: false }, init);

  const ring = assertRings(neither.series);
  const settled = assertSettles(both.series, neither.series);

  mkdirSync(path.dirname(OUT), { recursive: true });
  // Array order (reversed) sets the legend; drawOrder sets z-stacking. Dashed
  // Armijo-only draws above solid blue so their overlap at the EY gap stays visible.
  writeFileSync(
    OUT,
    renderSvg(
      [
        { series: neither.series, color: "#E69F00", width: 2, label: "neither", drawOrder: 0 },
        { series: armijoOnly.series, color: "#D55E00", width: 1.75, dash: "5 3", label: "Armijo only", drawOrder: 3 },
        { series: floorOnly.series, color: "#009E73", width: 1.75, dash: "5 3", label: "v† floor only", drawOrder: 1 },
        { series: both.series, color: "#0072B2", width: 2.25, label: "floor+Armijo", drawOrder: 2 },
      ],
      {
        ariaLabel:
          "Measured ‖Â_svd − Â_gd‖_F², 2×2 ablation over v† floor and Armijo backtracking",
        footer: `n=${N} k=${K} steps=${STEPS} lr=${BASE_LR} seed=${SEED} · raw (unprojected) gradient · generated by scripts/gen-ringing-plot.mjs`,
      },
    ),
  );

  const tail = (r) => r.series[r.series.length - 1].toExponential(2);
  console.log(
    `wrote ${OUT}\n` +
      `  metric: ‖Â_svd − Â_gd‖_F²\n` +
      `  neither rings: peaks=${ring.peaks} relSwing=${ring.relSwing.toFixed(2)}\n` +
      `  floored+Armijo settles: relSwing=${settled.relSwing.toFixed(3)} final=${settled.finalL.toExponential(3)}\n` +
      `  final gaps: both=${tail(both)} floorOnly=${tail(floorOnly)} armijoOnly=${tail(armijoOnly)} neither=${tail(neither)}\n` +
      `  Armijo effort: both mean α=2^${both.meanAlphaLog2.toFixed(2)} bt/step=${both.meanBt.toFixed(2)} ` +
      `(early ${both.meanBtEarly.toFixed(2)}) · armijoOnly mean α=2^${armijoOnly.meanAlphaLog2.toFixed(2)} ` +
      `bt/step=${armijoOnly.meanBt.toFixed(2)} (early ${armijoOnly.meanBtEarly.toFixed(2)})`,
  );
}

main();
