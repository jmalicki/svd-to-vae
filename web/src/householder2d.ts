/** Pure Householder / reflection helpers (2D geometry + nD column aiming). */

import {
  copy,
  fromNested,
  get,
  identity,
  mat,
  matmul,
  set,
  type Matrix,
} from "./matrix";

/**
 * Unit normal perpendicular to a mirror line at `lineAngleDeg` (degrees from +x).
 * Same convention as the mirror-angle sliders on the Factors chapter.
 */
export function normalFromLineAngle(lineAngleDeg: number): [number, number] {
  const nAng = ((lineAngleDeg + 90) * Math.PI) / 180;
  return [Math.cos(nAng), Math.sin(nAng)];
}

/** Split x into mirror-parallel and normal pieces for unit (or unnormalized) n. */
export function decomposeAlongNormal(
  x: number,
  y: number,
  nx: number,
  ny: number,
): {
  dot: number;
  parallel: [number, number];
  normal: [number, number];
} {
  const nlen = Math.hypot(nx, ny) || 1;
  const ux = nx / nlen;
  const uy = ny / nlen;
  const dot = x * ux + y * uy;
  const normal: [number, number] = [dot * ux, dot * uy];
  const parallel: [number, number] = [x - normal[0], y - normal[1]];
  return { dot, parallel, normal };
}

/** Unit normal (nx, ny) → Householder matrix H = I − 2 n nᵀ. */
export function householderFromNormal(nx: number, ny: number): Matrix {
  const n = Math.hypot(nx, ny) || 1;
  const x = nx / n;
  const y = ny / n;
  return fromNested([
    [1 - 2 * x * x, -2 * x * y],
    [-2 * x * y, 1 - 2 * y * y],
  ]);
}

/** Reflect point (x,y) across the line through the origin with unit normal (nx,ny). */
export function reflectAcrossNormal(
  x: number,
  y: number,
  nx: number,
  ny: number,
): [number, number] {
  const n = Math.hypot(nx, ny) || 1;
  const ux = nx / n;
  const uy = ny / n;
  const dot = 2 * (x * ux + y * uy);
  return [x - dot * ux, y - dot * uy];
}

/**
 * Householder that sends a = (ax, ay) to (±‖a‖, 0).
 * Default sign chooses the stable target (away from cancelling with a).
 */
export function householderAimToE1(
  ax: number,
  ay: number,
  sign?: 1 | -1,
): {
  H: Matrix;
  nx: number;
  ny: number;
  targetX: number;
  targetY: number;
  mirrorAngleDeg: number;
  normA: number;
} {
  const normA = Math.hypot(ax, ay);
  if (normA < 1e-15) {
    return {
      H: identity(2),
      nx: 1,
      ny: 0,
      targetX: 0,
      targetY: 0,
      mirrorAngleDeg: 0,
      normA: 0,
    };
  }
  // Teach with target on the +x-axis: (+‖a‖, 0).
  const s = sign ?? 1;
  const targetX = s * normA;
  const targetY = 0;
  // Mirror = perpendicular bisector of a and target ⇒ normal ∥ a − target.
  let nx = ax - targetX;
  let ny = ay - targetY;
  const nlen = Math.hypot(nx, ny);
  if (nlen < 1e-15) {
    // Already on the axis (same sign).
    return {
      H: identity(2),
      nx: 0,
      ny: 1,
      targetX,
      targetY,
      mirrorAngleDeg: 90,
      normA,
    };
  }
  nx /= nlen;
  ny /= nlen;
  const H = householderFromNormal(nx, ny);
  const mirrorAngleDeg = (Math.atan2(ny, nx) * 180) / Math.PI + 90;
  return { H, nx, ny, targetX, targetY, mirrorAngleDeg, normA };
}

/** Apply 2×2 matrix to a vector. */
export function applyMat2(M: Matrix, x: number, y: number): [number, number] {
  return [
    get(M, 0, 0) * x + get(M, 0, 1) * y,
    get(M, 1, 0) * x + get(M, 1, 1) * y,
  ];
}

/**
 * nD Householder reflector that maps column vector a to (±‖a‖, 0, …, 0).
 * Returns dense H (n×n).
 */
export function householderAimColumn(a: number[], sign?: 1 | -1): Matrix {
  const n = a.length;
  const H = identity(n);
  const normA = Math.hypot(...a);
  if (normA < 1e-15) return H;
  const s = sign ?? 1;
  const target0 = s * normA;
  const v = new Float64Array(n);
  v[0] = a[0] - target0;
  for (let i = 1; i < n; i++) v[i] = a[i];
  let vTv = 0;
  for (let i = 0; i < n; i++) vTv += v[i] * v[i];
  if (vTv < 1e-30) return H;
  const scale = 2 / vTv;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      set(H, i, j, get(H, i, j) - scale * v[i] * v[j]);
    }
  }
  return H;
}

export function columnOf(A: Matrix, j: number): number[] {
  const col: number[] = [];
  for (let i = 0; i < A.rows; i++) col.push(get(A, i, j));
  return col;
}

export function rowOf(A: Matrix, i: number): number[] {
  const row: number[] = [];
  for (let j = 0; j < A.cols; j++) row.push(get(A, i, j));
  return row;
}

/** Left-multiply: H A. */
export function applyLeft(H: Matrix, A: Matrix): Matrix {
  return matmul(H, A);
}

/** Right-multiply: A H. */
export function applyRight(A: Matrix, H: Matrix): Matrix {
  return matmul(A, H);
}

/**
 * Givens rotation (c, s) acting on coordinates (i, k): plane rotation with det +1.
 * For 2D zeroing of y-component of (x,y): c = x/r, s = y/r maps to (r, 0).
 */
export function givensAimToE1(x: number, y: number): Matrix {
  const r = Math.hypot(x, y);
  if (r < 1e-15) return identity(2);
  const c = x / r;
  const s = y / r;
  // Rᵀ [x,y] = [r, 0] with R = [[c,s],[-s,c]]? 
  // [[c,-s],[s,c]] [x,y] if we want rotation... 
  // We want Q such that Q [x,y]^T = [r,0]^T with Q orthogonal det+1:
  // Q = [[c, s], [-s, c]] where c=x/r, s=y/r: Q[x,y]=[c x + s y, -s x + c y]=[r, 0].
  return fromNested([
    [c, s],
    [-s, c],
  ]);
}

/** After left Householder on 2×2, apply right Givens to zero the (0,1) entry when possible. */
export function rightGivensZeroSuperdiag(A: Matrix): Matrix {
  // A is expected roughly [[α, β], [0, γ]]. Mix columns to zero β.
  const alpha = get(A, 0, 0);
  const beta = get(A, 0, 1);
  const r = Math.hypot(alpha, beta);
  if (r < 1e-15) return identity(2);
  const c = alpha / r;
  const s = beta / r;
  // A [[c, -s], [s, c]] — first column becomes (r, something aligned)
  // [[α,β],[0,γ]] [[c,-s],[s,c]] = [[αc+βs, -αs+βc],[γs, γc]]
  // Want -αs+βc = 0 ⇒ s/c = β/α ⇒ s = β/r, c = α/r. Yes.
  return fromNested([
    [c, -s],
    [s, c],
  ]);
}

export function lerp2(
  a: [number, number],
  b: [number, number],
  t: number,
): [number, number] {
  return [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t];
}

/** Build a small demo matrix with a nontrivial first column. */
export function demoMatrix2(): Matrix {
  return fromNested([
    [1.5, 1.0],
    [0.8, 1.2],
  ]);
}

export function takeColumnMatrix(A: Matrix, j: number): Matrix {
  const c = mat(A.rows, 1);
  for (let i = 0; i < A.rows; i++) set(c, i, 0, get(A, i, j));
  return c;
}

export { copy, identity, mat, get, set, fromNested, matmul };
