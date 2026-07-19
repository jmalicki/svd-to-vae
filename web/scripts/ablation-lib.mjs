/**
 * Shared training loop and plot helpers for the build-time optimizer ablations.
 *
 * One loop, three independent toggles — {v† floor, Armijo, tangent projection} —
 * so gen-ringing-plot.mjs (floor × Armijo) and gen-freeze-plot.mjs
 * (projection × Armijo, plus the full 2×2×2 cube) measure the exact same
 * dynamics and cannot drift apart. Mirrors web/src/svdGrad.ts, including the
 * page's init order (U, then raw σ, then V from one mulberry32 stream).
 */
import { Matrix as MlMatrix, SingularValueDecomposition } from "ml-matrix";

export const BETA2 = 0.999;
export const ARMIJO_C = 1e-4;
export const ARMIJO_MAX_BT = 12;

// --- tiny RNG / matrix helpers (mirrors web/src/matrix.ts) ---

export function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function mat(rows, cols, fill = 0) {
  return { rows, cols, data: new Float64Array(rows * cols).fill(fill) };
}

export function get(m, i, j) {
  return m.data[i * m.cols + j];
}

export function set(m, i, j, v) {
  m.data[i * m.cols + j] = v;
}

export function copy(m) {
  return { rows: m.rows, cols: m.cols, data: new Float64Array(m.data) };
}

export function transpose(m) {
  const t = mat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) set(t, j, i, get(m, i, j));
  }
  return t;
}

export function matmul(a, b) {
  const c = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let p = 0; p < a.cols; p++) {
      const aik = get(a, i, p);
      for (let j = 0; j < b.cols; j++) c.data[i * c.cols + j] += aik * get(b, p, j);
    }
  }
  return c;
}

export function frobeniusSq(m) {
  let s = 0;
  for (let i = 0; i < m.data.length; i++) s += m.data[i] * m.data[i];
  return s;
}

export function maxAbs(m) {
  let mabs = 0;
  for (let i = 0; i < m.data.length; i++) mabs = Math.max(mabs, Math.abs(m.data[i]));
  return mabs || 1;
}

export function thinQ(m) {
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

export function randn(rand, rows, cols, scale) {
  const m = mat(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    m.data[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return m;
}

export function softplus(x) {
  return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
}

export function softplusPrime(x) {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

export function reconstruct(U, sigma, V) {
  const US = copy(U);
  for (let j = 0; j < sigma.length; j++) {
    for (let i = 0; i < U.rows; i++) set(US, i, j, get(U, i, j) * sigma[j]);
  }
  return matmul(US, transpose(V));
}

export function truncSvd(A, k) {
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

export function diffFrobeniusSq(A, B) {
  let s = 0;
  for (let i = 0; i < A.data.length; i++) {
    const d = A.data[i] - B.data[i];
    s += d * d;
  }
  return s;
}

export function gradSecondMomentFpFloor(dataScale) {
  const ulp = Number.EPSILON * Math.max(dataScale, Number.EPSILON);
  return ulp * ulp;
}

/** g − X·sym(Xᵀg): project a factor gradient onto the Stiefel tangent space at X. */
export function stiefelProject(X, g) {
  const XtG = matmul(transpose(X), g);
  const k = XtG.cols;
  const sym = mat(k, k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      sym.data[i * k + j] = 0.5 * (XtG.data[i * k + j] + XtG.data[j * k + i]);
    }
  }
  const XSym = matmul(X, sym);
  const out = copy(g);
  for (let i = 0; i < out.data.length; i++) out.data[i] -= XSym.data[i];
  return out;
}

/**
 * Analytical grads of MSE = mean((A − U diag(σ) Vᵀ)²), σ = softplus(raw).
 */
export function grads(A, U, raw, V) {
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

function addScaledMat(X, G, scale) {
  const out = copy(X);
  for (let i = 0; i < out.data.length; i++) out.data[i] += scale * G.data[i];
  return out;
}

/**
 * Draw the exact init the gradient page uses for a given seed: N(0, 0.3²)
 * entries for U, then raw σ, then V, from one mulberry32(seed ^ 0x9e3779b9)
 * stream, then thin-QR on U and V (see SvdGradTrainer.init / gradient.ts).
 */
export function pageInit(seed, n, k) {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const U = randn(rand, n, k, 0.3);
  const rawM = randn(rand, 1, k, 0.3);
  const V = randn(rand, n, k, 0.3);
  return { U: thinQ(U), raw: Float64Array.from(rawM.data), V: thinQ(V) };
}

/** The page's matrix for a given seed: n×n standard normal from mulberry32(seed). */
export function pageMatrix(seed, n) {
  return randn(mulberry32(seed), n, n, 1);
}

/**
 * Global-RMS SGD + thin-QR retraction with three independent toggles:
 *   floor   — v† floor on the second moment (vs bias-corrected EMA + 1e-8)
 *   armijo  — backtracking line search on the retracted trial point
 *   project — Stiefel tangent projection of the U/V gradients before stepping
 * Returns the ‖Â_svd − Â_gd‖² series plus line-search/reject statistics.
 */
export function train(A, Asvd, { k, steps, baseLr, floor, armijo, project }, init) {
  let U = copy(init.U);
  let V = copy(init.V);
  let raw = Float64Array.from(init.raw);

  const vFp = gradSecondMomentFpFloor(maxAbs(A));
  let vMom = 0;
  const vsSvd = [];
  let btTotal = 0;
  let btEarly = 0;
  let alphaLogSum = 0;
  let armijoSteps = 0;
  const tailWindow = Math.min(200, Math.floor(steps / 4));
  let tailRejects = 0;

  for (let t = 1; t <= steps; t++) {
    const g = grads(A, U, Array.from(raw), V);
    const gU = project ? stiefelProject(U, g.gU) : g.gU;
    const gV = project ? stiefelProject(V, g.gV) : g.gV;
    const gRaw = g.gRaw;
    vsSvd.push(diffFrobeniusSq(Asvd, g.Ahat));

    let sumSq = 0;
    let count = 0;
    for (let i = 0; i < gU.data.length; i++) {
      sumSq += gU.data[i] * gU.data[i];
      count++;
    }
    for (let i = 0; i < gV.data.length; i++) {
      sumSq += gV.data[i] * gV.data[i];
      count++;
    }
    for (let i = 0; i < gRaw.length; i++) {
      sumSq += gRaw[i] * gRaw[i];
      count++;
    }
    const meanSq = sumSq / count;

    vMom = BETA2 * vMom + (1 - BETA2) * meanSq;
    const vHat = vMom / (1 - BETA2 ** t);
    let denom;
    if (floor) {
      const vUse = Math.max(vHat, meanSq, vFp);
      denom = Math.sqrt(vUse) + Math.sqrt(vFp);
    } else {
      // Unfloored global RMS (the failure mode): only bias-corrected EMA + tiny ε.
      denom = Math.sqrt(Math.max(vHat, 1e-30)) + 1e-8;
    }
    const eta = baseLr / denom;
    const mse0 = mseFrom(A, U, raw, V);
    const dirDeriv = -eta * sumSq;

    if (!armijo) {
      for (let i = 0; i < U.data.length; i++) U.data[i] -= eta * gU.data[i];
      for (let i = 0; i < V.data.length; i++) V.data[i] -= eta * gV.data[i];
      for (let j = 0; j < k; j++) raw[j] -= eta * gRaw[j];
      U = thinQ(U);
      V = thinQ(V);
    } else {
      let alpha = 1;
      let chosen = null;
      let bestDescent = null;
      armijoSteps++;
      for (let bt = 0; bt < ARMIJO_MAX_BT; bt++) {
        btTotal++;
        if (t <= steps / 3) btEarly++;
        const Utry = thinQ(addScaledMat(U, gU, -alpha * eta));
        const Vtry = thinQ(addScaledMat(V, gV, -alpha * eta));
        const rawTry = Float64Array.from(raw, (r, i) => r - alpha * eta * gRaw[i]);
        const mseTry = mseFrom(A, Utry, rawTry, Vtry);
        if (!bestDescent || mseTry < bestDescent.L) {
          bestDescent = { U: Utry, V: Vtry, raw: rawTry, L: mseTry, alpha };
        }
        if (mseTry <= mse0 + ARMIJO_C * alpha * dirDeriv) {
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
      } else if (t > steps - tailWindow) {
        tailRejects++;
      }
    }
  }

  const sigma = Array.from(raw, softplus);
  const finalGap = diffFrobeniusSq(Asvd, reconstruct(U, sigma, V));

  return {
    series: vsSvd,
    finalGap,
    tailRejects,
    tailWindow,
    meanAlphaLog2: armijoSteps ? alphaLogSum / armijoSteps : 0,
    meanBt: armijoSteps ? btTotal / armijoSteps : 0,
    meanBtEarly: armijoSteps ? btEarly / Math.floor(steps / 3) : 0,
  };
}

// --- SVG rendering (shared by both ablation figures) ---

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

/**
 * Render log-scale gap-vs-step curves. `runs`: array of
 * { series, color, width, label, drawOrder, dash? } — array order (reversed)
 * sets the legend; drawOrder sets z-stacking. `meta`: { ariaLabel, footer }.
 */
export function renderSvg(runs, meta) {
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
  role="img" aria-label="${meta.ariaLabel}">
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
    font-family="IBM Plex Mono, ui-monospace, monospace">${meta.footer}</text>
</svg>
`;
}
