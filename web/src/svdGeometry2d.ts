/** Pure 2×2 SVD / ellipse helpers for the geometry chapter (no DOM). */

import { classicalSvd } from "./classicalSvd";
import {
  fromNested,
  get,
  matmul,
  transpose,
  type Matrix,
} from "./matrix";

export type AngleFactors = {
  s1: number;
  s2: number;
  thVDeg: number;
  thUDeg: number;
};

/** Stages used by the rotate → stretch → rotate movie. */
export type EllipseFrame = {
  A: Matrix;
  U: Matrix;
  V: Matrix;
  Vt: Matrix;
  S: Matrix;
  /** Σ Vᵀ — intermediate after stretch */
  SV: Matrix;
  sigma: [number, number];
  outScale: number;
};

/** General (tilted) example: σ ≠ diagonal of A. */
export const TILTED_EXAMPLE_A = fromNested([
  [1.5, 1.0],
  [0.2, 1.2],
]);

export function degToRad(d: number): number {
  return (d * Math.PI) / 180;
}

/** 2D rotation matrix [[c,-s],[s,c]]. */
export function rot2(thetaRad: number): Matrix {
  const c = Math.cos(thetaRad);
  const s = Math.sin(thetaRad);
  return fromNested([
    [c, -s],
    [s, c],
  ]);
}

/** 2D rotation by θ degrees (counterclockwise). */
export function rotationMatrixDeg(deg: number): Matrix {
  return rot2(degToRad(deg));
}

/** Axis-aligned stretch: scales x by sx and y by sy. */
export function stretchMatrix(sx: number, sy: number): Matrix {
  return fromNested([
    [sx, 0],
    [0, sy],
  ]);
}

export function applyMat2(M: Matrix, x: number, y: number): [number, number] {
  return [
    get(M, 0, 0) * x + get(M, 0, 1) * y,
    get(M, 1, 0) * x + get(M, 1, 1) * y,
  ];
}

export function vecLen(x: number, y: number): number {
  return Math.hypot(x, y);
}

/**
 * Build A = U diag(σ) Vᵀ from stretch amounts and rotation angles (degrees).
 * U and V are pure rotations (det = +1).
 */
export function buildAFromFactors(f: AngleFactors): {
  A: Matrix;
  U: Matrix;
  V: Matrix;
  sigma: [number, number];
} {
  const U = rot2(degToRad(f.thUDeg));
  const V = rot2(degToRad(f.thVDeg));
  const S = stretchMatrix(f.s1, f.s2);
  const A = matmul(matmul(U, S), transpose(V));
  return { A, U, V, sigma: [f.s1, f.s2] };
}

function outScaleFor(s1: number): number {
  return Math.max(2.4, s1 * 1.25, 1.2);
}

function packFrame(
  A: Matrix,
  U: Matrix,
  V: Matrix,
  s1: number,
  s2: number,
): EllipseFrame {
  const Vt = transpose(V);
  const S = stretchMatrix(s1, s2);
  return {
    A,
    U,
    V,
    Vt,
    S,
    SV: matmul(S, Vt),
    sigma: [s1, s2],
    outScale: outScaleFor(s1),
  };
}

/** SVD of A plus intermediate matrices for the four-panel movie. */
export function frameFromMatrix(A: Matrix): EllipseFrame {
  const { U, V, sigma } = classicalSvd(A, 2);
  return packFrame(A, U, V, sigma[0] ?? 0, sigma[1] ?? 0);
}

/** Same frame shape, built from scrubbable factors (synthesis playground). */
export function frameFromFactors(f: AngleFactors): EllipseFrame {
  const { A, U, V, sigma } = buildAFromFactors(f);
  return packFrame(A, U, V, sigma[0], sigma[1]);
}

/**
 * Sample ‖Ax‖ over the unit circle. For a full-rank 2×2 map the extrema are σ₁, σ₂.
 */
export function sampleStretchRange(
  A: Matrix,
  samples = 360,
): { max: number; min: number } {
  let max = 0;
  let min = Number.POSITIVE_INFINITY;
  for (let i = 0; i < samples; i++) {
    const t = (i / samples) * Math.PI * 2;
    const [ax, ay] = applyMat2(A, Math.cos(t), Math.sin(t));
    const len = vecLen(ax, ay);
    if (len > max) max = len;
    if (len < min) min = len;
  }
  return { max, min };
}

/** Apply movie stages to a unit vector: x → Vᵀx → ΣVᵀx → UΣVᵀx. */
export function movieStages(
  f: EllipseFrame,
  x: number,
  y: number,
): {
  afterVt: [number, number];
  afterS: [number, number];
  afterU: [number, number];
  direct: [number, number];
} {
  const afterVt = applyMat2(f.Vt, x, y);
  const afterS = applyMat2(f.SV, x, y);
  const afterU = applyMat2(f.U, afterS[0], afterS[1]);
  const direct = applyMat2(f.A, x, y);
  return { afterVt, afterS, afterU, direct };
}
