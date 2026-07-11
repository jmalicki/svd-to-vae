import { describe, expect, it } from "vitest";
import {
  frobeniusSq,
  get,
  mat,
  matmul,
  reconstruct,
  thinQ,
  transpose,
} from "./matrix";
import { gradSecondMomentFpFloor } from "./svdGrad";

/** Full-column-rank matrix so thin QR is well-conditioned. */
function tallFullRank(rows: number, cols: number) {
  const A = mat(rows, cols);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      A.data[i * cols + j] = (i === j ? 1 : 0) + 0.05 * Math.sin(i + 2 * j + 1);
    }
  }
  return A;
}

describe("thinQ", () => {
  it("returns orthonormal columns (QᵀQ ≈ I)", () => {
    const Q = thinQ(tallFullRank(6, 3));
    const QtQ = matmul(transpose(Q), Q);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(get(QtQ, i, j)).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
  });
});

describe("reconstruct", () => {
  it("satisfies ‖U diag(σ) Vᵀ‖_F² = ∑ σⱼ² for orthonormal factors", () => {
    const U = thinQ(tallFullRank(4, 2));
    const V = thinQ(tallFullRank(3, 2));
    const sigma = [2, 0.5];
    const Ahat = reconstruct(U, sigma, V);
    expect(frobeniusSq(Ahat)).toBeCloseTo(2 * 2 + 0.5 * 0.5, 10);
  });
});

describe("gradSecondMomentFpFloor", () => {
  it("is (ε · scale)²", () => {
    expect(gradSecondMomentFpFloor(1)).toBeCloseTo(Number.EPSILON ** 2, 30);
    expect(gradSecondMomentFpFloor(10)).toBeCloseTo((10 * Number.EPSILON) ** 2, 30);
  });
});
