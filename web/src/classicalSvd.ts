import { Matrix as MlMatrix, SingularValueDecomposition } from "ml-matrix";
import { type Matrix, fromNested, takeColumns } from "./matrix";

export type SvdResult = {
  U: Matrix;
  sigma: number[];
  V: Matrix;
};

/**
 * Truncated classical SVD via ml-matrix.
 * Returns thin factors with at most `rank` components (singular values descending).
 */
export function classicalSvd(A: Matrix, rank: number): SvdResult {
  const fullK = Math.min(A.rows, A.cols);
  const k = Math.max(1, Math.min(rank, fullK));

  const M = new MlMatrix(toRows(A));
  const svd = new SingularValueDecomposition(M, { autoTranspose: true });

  const Ufull = fromNested(svd.leftSingularVectors.to2DArray());
  const Vfull = fromNested(svd.rightSingularVectors.to2DArray());
  const sigmaFull = svd.diagonal.slice();

  // Fix signs for stable visualization: largest-abs entry in each U column >= 0
  const U = takeColumns(Ufull, k);
  const V = takeColumns(Vfull, k);
  const sigma = sigmaFull.slice(0, k);

  for (let j = 0; j < k; j++) {
    let best = 0;
    let bestAbs = 0;
    for (let i = 0; i < U.rows; i++) {
      const v = U.data[i * U.cols + j];
      if (Math.abs(v) > bestAbs) {
        bestAbs = Math.abs(v);
        best = v;
      }
    }
    if (best < 0) {
      for (let i = 0; i < U.rows; i++) U.data[i * U.cols + j] *= -1;
      for (let i = 0; i < V.rows; i++) V.data[i * V.cols + j] *= -1;
    }
  }

  return { U, sigma, V };
}

function toRows(A: Matrix): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < A.rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < A.cols; j++) row.push(A.data[i * A.cols + j]);
    rows.push(row);
  }
  return rows;
}
