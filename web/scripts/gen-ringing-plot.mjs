/**
 * Build-time ablation: global-RMS SGD, 2×2 over {v† floor on/off} × {Armijo on/off}.
 * Plots ‖Â_svd − Â_gd‖_F² (the amber curve in the live demo) for all four runs.
 * Writes public/ringing-floor.svg and exits nonzero if the neither-fix run does not
 * ring or the floored+Armijo run fails to stay near the Eckart–Young gap.
 *
 * Run: node scripts/gen-ringing-plot.mjs
 */
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Matrix as MlMatrix, SingularValueDecomposition } from "ml-matrix";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, "../public/ringing-floor.svg");

const BETA2 = 0.999;
const N = 6;
const K = 3;
const STEPS = 900;
const BASE_LR = 0.05;
const SEED = 7;

// --- tiny RNG / matrix helpers (mirrors web/src/matrix.ts) ---

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mat(rows, cols, fill = 0) {
  return { rows, cols, data: new Float64Array(rows * cols).fill(fill) };
}

function get(m, i, j) {
  return m.data[i * m.cols + j];
}

function set(m, i, j, v) {
  m.data[i * m.cols + j] = v;
}

function copy(m) {
  return { rows: m.rows, cols: m.cols, data: new Float64Array(m.data) };
}

function transpose(m) {
  const t = mat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) set(t, j, i, get(m, i, j));
  }
  return t;
}

function matmul(a, b) {
  const c = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let p = 0; p < a.cols; p++) {
      const aik = get(a, i, p);
      for (let j = 0; j < b.cols; j++) c.data[i * c.cols + j] += aik * get(b, p, j);
    }
  }
  return c;
}

function frobeniusSq(m) {
  let s = 0;
  for (let i = 0; i < m.data.length; i++) s += m.data[i] * m.data[i];
  return s;
}

function maxAbs(m) {
  let mabs = 0;
  for (let i = 0; i < m.data.length; i++) mabs = Math.max(mabs, Math.abs(m.data[i]));
  return mabs || 1;
}

function thinQ(m) {
  const Q = copy(m);
  const { rows: n, cols: k } = Q;
  for (let j = 0; j < k; j++) {
    let normSq = 0;
    for (let i = 0; i < n; i++) normSq += get(Q, i, j) ** 2;
    const inv = 1 / Math.sqrt(Math.max(normSq, 1e-30));
    for (let i = 0; i < n; i++) set(Q, i, j, get(Q, i, j) * inv);
    for (let ell = j + 1; ell < k; ell++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += get(Q, i, j) * get(Q, i, ell);
      for (let i = 0; i < n; i++) set(Q, i, ell, get(Q, i, ell) - dot * get(Q, i, j));
    }
  }
  return Q;
}

function randn(rand, rows, cols, scale) {
  const m = mat(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    m.data[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return m;
}

function softplus(x) {
  return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
}

function softplusPrime(x) {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

function reconstruct(U, sigma, V) {
  const US = copy(U);
  for (let j = 0; j < sigma.length; j++) {
    for (let i = 0; i < U.rows; i++) set(US, i, j, get(U, i, j) * sigma[j]);
  }
  return matmul(US, transpose(V));
}

function truncSvd(A, k) {
  const rows = [];
  for (let i = 0; i < A.rows; i++) {
    const row = [];
    for (let j = 0; j < A.cols; j++) row.push(get(A, i, j));
    rows.push(row);
  }
  const svd = new SingularValueDecomposition(new MlMatrix(rows), { autoTranspose: true });
  const s = svd.diagonal;
  const U = svd.leftSingularVectors.to2DArray();
  const V = svd.rightSingularVectors.to2DArray();
  const Um = mat(A.rows, k);
  const Vm = mat(A.cols, k);
  const sigma = [];
  for (let j = 0; j < k; j++) {
    sigma.push(s[j]);
    for (let i = 0; i < A.rows; i++) set(Um, i, j, U[i][j]);
    for (let i = 0; i < A.cols; i++) set(Vm, i, j, V[i][j]);
  }
  return reconstruct(Um, sigma, Vm);
}

function diffFrobeniusSq(A, B) {
  let s = 0;
  for (let i = 0; i < A.data.length; i++) {
    const d = A.data[i] - B.data[i];
    s += d * d;
  }
  return s;
}

function gradSecondMomentFpFloor(dataScale) {
  const ulp = Number.EPSILON * Math.max(dataScale, Number.EPSILON);
  return ulp * ulp;
}

/**
 * Analytical grads of MSE = mean((A − U diag(σ) Vᵀ)²), σ = softplus(raw).
 */
function grads(A, U, raw, V) {
  const m = A.rows;
  const n = A.cols;
  const k = raw.length;
  const sigma = raw.map(softplus);
  const Ahat = reconstruct(U, sigma, V);
  const R = mat(m, n);
  for (let i = 0; i < R.data.length; i++) R.data[i] = Ahat.data[i] - A.data[i];
  const scale = 2 / (m * n);

  const gU = mat(m, k);
  const gV = mat(n, k);
  const gRaw = new Float64Array(k);

  for (let j = 0; j < k; j++) {
    let dSigma = 0;
    for (let i = 0; i < m; i++) {
      let acc = 0;
      for (let t = 0; t < n; t++) acc += get(R, i, t) * get(V, t, j);
      set(gU, i, j, scale * acc * sigma[j]);
      dSigma += get(U, i, j) * acc;
    }
    for (let t = 0; t < n; t++) {
      let acc = 0;
      for (let i = 0; i < m; i++) acc += get(R, i, t) * get(U, i, j);
      set(gV, t, j, scale * acc * sigma[j]);
    }
    gRaw[j] = scale * dSigma * softplusPrime(raw[j]);
  }

  return { gU, gV, gRaw, Ahat };
}

function meanGradSq(gU, gRaw, gV) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < gU.data.length; i++) {
    sum += gU.data[i] * gU.data[i];
    count++;
  }
  for (let i = 0; i < gV.data.length; i++) {
    sum += gV.data[i] * gV.data[i];
    count++;
  }
  for (let i = 0; i < gRaw.length; i++) {
    sum += gRaw[i] * gRaw[i];
    count++;
  }
  return sum / count;
}

function addScaledMat(X, G, scale) {
  const out = copy(X);
  for (let i = 0; i < out.data.length; i++) out.data[i] += scale * G.data[i];
  return out;
}

function mseFrom(A, U, raw, V) {
  const sigma = Array.from(raw, softplus);
  const Ahat = reconstruct(U, sigma, V);
  let s = 0;
  for (let i = 0; i < A.data.length; i++) {
    const d = Ahat.data[i] - A.data[i];
    s += d * d;
  }
  return s / (A.rows * A.cols);
}

/** Returns series of ‖Â_svd − Â_gd‖_F² after each step. */
function train(A, Asvd, { floor, armijo }, seed) {
  const rand = mulberry32(seed);
  const m = A.rows;
  const n = A.cols;
  let U = thinQ(randn(rand, m, K, 0.3));
  let V = thinQ(randn(rand, n, K, 0.3));
  let raw = new Float64Array(K);
  for (let j = 0; j < K; j++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    raw[j] = 0.3 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const vFp = gradSecondMomentFpFloor(maxAbs(A));
  let vMom = 0;
  const vsSvd = [];
  let btTotal = 0;
  let btEarly = 0;
  let alphaLogSum = 0;
  let armijoSteps = 0;

  for (let t = 1; t <= STEPS; t++) {
    const { gU, gV, gRaw, Ahat } = grads(A, U, Array.from(raw), V);
    vsSvd.push(diffFrobeniusSq(Asvd, Ahat));

    const meanSq = meanGradSq(gU, gRaw, gV);
    vMom = BETA2 * vMom + (1 - BETA2) * meanSq;
    const vHat = vMom / (1 - BETA2 ** t);
    let vUse;
    let denom;
    if (floor) {
      vUse = Math.max(vHat, meanSq, vFp);
      denom = Math.sqrt(vUse) + Math.sqrt(vFp);
    } else {
      // Unfloored global RMS (the failure mode): only bias-corrected EMA + tiny ε.
      vUse = Math.max(vHat, 1e-30);
      denom = Math.sqrt(vUse) + 1e-8;
    }
    const eta = BASE_LR / denom;
    const mse0 = (() => {
      let s = 0;
      for (let i = 0; i < A.data.length; i++) {
        const d = Ahat.data[i] - A.data[i];
        s += d * d;
      }
      return s / (A.rows * A.cols);
    })();
    const sumSq = meanSq * (gU.data.length + gV.data.length + gRaw.length);
    const dirDeriv = -eta * sumSq;

    const maxBt = armijo ? 12 : 0;
    if (maxBt === 0) {
      for (let i = 0; i < U.data.length; i++) U.data[i] -= eta * gU.data[i];
      for (let i = 0; i < V.data.length; i++) V.data[i] -= eta * gV.data[i];
      for (let j = 0; j < K; j++) raw[j] -= eta * gRaw[j];
      U = thinQ(U);
      V = thinQ(V);
    } else {
      let alpha = 1;
      let chosen = null;
      let bestDescent = null;
      armijoSteps++;
      for (let bt = 0; bt < maxBt; bt++) {
        btTotal++;
        if (t <= STEPS / 3) btEarly++;
        const Utry = thinQ(addScaledMat(U, gU, -alpha * eta));
        const Vtry = thinQ(addScaledMat(V, gV, -alpha * eta));
        const rawTry = Float64Array.from(raw, (r, i) => r - alpha * eta * gRaw[i]);
        const mseTry = mseFrom(A, Utry, rawTry, Vtry);
        if (!bestDescent || mseTry < bestDescent.L) {
          bestDescent = { U: Utry, V: Vtry, raw: rawTry, L: mseTry, alpha };
        }
        if (mseTry <= mse0 + 1e-4 * alpha * dirDeriv) {
          chosen = { U: Utry, V: Vtry, raw: rawTry, alpha };
          break;
        }
        alpha *= 0.5;
      }
      if (!chosen && bestDescent && bestDescent.L < mse0) {
        chosen = bestDescent;
      }
      if (chosen) {
        U = chosen.U;
        V = chosen.V;
        raw = chosen.raw;
        alphaLogSum += Math.log2(chosen.alpha);
      }
    }
  }

  return {
    series: vsSvd,
    meanAlphaLog2: armijoSteps ? alphaLogSum / armijoSteps : 0,
    meanBt: armijoSteps ? btTotal / armijoSteps : 0,
    meanBtEarly: armijoSteps ? btEarly / Math.floor(STEPS / 3) : 0,
  };
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

function toPath(series, x0, x1, y0, y1, yLog) {
  const n = series.length;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = x0 + ((x1 - x0) * i) / (n - 1);
    const L = Math.max(series[i], 1e-18);
    const t = (Math.log10(L) - yLog.min) / (yLog.max - yLog.min);
    const y = y1 - t * (y1 - y0);
    pts.push(`${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return pts.join(" ");
}

function renderSvg(runs) {
  const all = runs.flatMap((r) => r.series).concat([1e-18]);
  let lo = Infinity;
  let hi = -Infinity;
  for (const L of all) {
    const v = Math.log10(Math.max(L, 1e-18));
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const pad = 0.08 * (hi - lo || 1);
  const yLog = { min: lo - pad, max: hi + pad };

  const W = 560;
  const H = 232;
  const x0 = 56;
  const x1 = 540;
  const y0 = 40;
  const y1 = 187;

  const paths = runs
    .slice()
    .sort((a, b) => a.drawOrder - b.drawOrder)
    .map(
      (r) =>
        `  <path d="${toPath(r.series, x0, x1, y0, y1, yLog)}" fill="none" ` +
        `stroke="${r.color}" stroke-width="${r.width}"${r.dash ? ` stroke-dasharray="${r.dash}"` : ""}/>`,
    )
    .join("\n");

  let legendX = x0;
  const legend = runs
    .slice()
    .reverse()
    .map((r) => {
      const lineX = legendX;
      const textX = legendX + 26;
      legendX += 26 + 7 * r.label.length + 16;
      return (
        `  <line x1="${lineX}" y1="18" x2="${lineX + 20}" y2="18" stroke="${r.color}" ` +
        `stroke-width="${r.width}"${r.dash ? ` stroke-dasharray="${r.dash}"` : ""}/>\n` +
        `  <text x="${textX}" y="22" fill="#2d2d2d" font-size="11" font-family="DM Sans, system-ui, sans-serif">${r.label}</text>`
      );
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"
  role="img" aria-label="Measured ‖Â_svd − Â_gd‖_F², 2×2 ablation over v† floor and Armijo backtracking">
  <rect width="${W}" height="${H}" fill="#fafafa"/>
  <line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" stroke="#2d2d2d" stroke-width="1.25"/>
  <line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="#2d2d2d" stroke-width="1.25"/>
  <text x="16" y="130" fill="#6b6b6b" font-size="11" font-family="IBM Plex Mono, ui-monospace, monospace"
    transform="rotate(-90 16 130)">‖Â_svd−Â_gd‖² (log)</text>
  <text x="${(x0 + x1) / 2}" y="203" fill="#6b6b6b" font-size="11"
    font-family="IBM Plex Mono, ui-monospace, monospace" text-anchor="middle">step</text>
${paths}
${legend}
  <text x="${x0}" y="${H - 6}" fill="#8a8a8a" font-size="9"
    font-family="IBM Plex Mono, ui-monospace, monospace">n=${N} k=${K} steps=${STEPS} lr=${BASE_LR} seed=${SEED} · generated by scripts/gen-ringing-plot.mjs</text>
</svg>
`;
}

function main() {
  const rand = mulberry32(SEED);
  const A = randn(rand, N, N, 1);
  const Asvd = truncSvd(A, K);

  const both = train(A, Asvd, { floor: true, armijo: true }, SEED + 1);
  const floorOnly = train(A, Asvd, { floor: true, armijo: false }, SEED + 1);
  const armijoOnly = train(A, Asvd, { floor: false, armijo: true }, SEED + 1);
  const neither = train(A, Asvd, { floor: false, armijo: false }, SEED + 1);

  const ring = assertRings(neither.series);
  const settled = assertSettles(both.series, neither.series);

  mkdirSync(path.dirname(OUT), { recursive: true });
  // Array order (reversed) sets the legend; drawOrder sets z-stacking. Dashed
  // Armijo-only draws above solid blue so their overlap at the EY gap stays visible.
  writeFileSync(
    OUT,
    renderSvg([
      { series: neither.series, color: "#E69F00", width: 2, label: "neither", drawOrder: 0 },
      { series: armijoOnly.series, color: "#D55E00", width: 1.75, dash: "5 3", label: "Armijo only", drawOrder: 3 },
      { series: floorOnly.series, color: "#009E73", width: 1.75, dash: "5 3", label: "v† floor only", drawOrder: 1 },
      { series: both.series, color: "#0072B2", width: 2.25, label: "floor+Armijo", drawOrder: 2 },
    ]),
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
