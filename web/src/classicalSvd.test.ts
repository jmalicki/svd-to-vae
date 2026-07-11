import { describe, expect, it } from "vitest";
import { classicalSvd } from "./classicalSvd";
import { frobeniusSq, mat, reconstruct, sub } from "./matrix";

describe("classicalSvd", () => {
  it("matches Eckart–Young: truncated recon error is ‖A‖_F² − ∑_{j≤k} σⱼ²", () => {
    const A = mat(4, 4);
    for (let i = 0; i < A.data.length; i++) A.data[i] = (i % 5) * 0.2 - 0.4;
    const k = 2;
    const svd = classicalSvd(A, k);
    const recon = reconstruct(svd.U, svd.sigma, svd.V);
    const err = frobeniusSq(sub(A, recon));
    const full = classicalSvd(A, 4);
    const expected =
      frobeniusSq(A) - full.sigma.slice(0, k).reduce((s, x) => s + x * x, 0);
    expect(err).toBeCloseTo(expected, 8);
    expect(svd.sigma.length).toBe(k);
    for (let j = 1; j < k; j++) {
      expect(svd.sigma[j - 1]!).toBeGreaterThanOrEqual(svd.sigma[j]!);
    }
  });
});
