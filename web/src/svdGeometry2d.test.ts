import { describe, expect, it } from "vitest";
import { classicalSvd } from "./classicalSvd";
import { frobeniusSq, fromNested, get, sub } from "./matrix";
import {
  TILTED_EXAMPLE_A,
  applyMat2,
  buildAFromFactors,
  frameFromFactors,
  frameFromMatrix,
  movieStages,
  rot2,
  sampleStretchRange,
  vecLen,
} from "./svdGeometry2d";

describe("rot2", () => {
  it("is orthogonal with det +1", () => {
    const R = rot2(0.7);
    // Rᵀ R ≈ I
    const c = Math.cos(0.7);
    const s = Math.sin(0.7);
    expect(get(R, 0, 0)).toBeCloseTo(c, 12);
    expect(get(R, 0, 1)).toBeCloseTo(-s, 12);
    expect(get(R, 1, 0)).toBeCloseTo(s, 12);
    expect(get(R, 1, 1)).toBeCloseTo(c, 12);
    const det = get(R, 0, 0) * get(R, 1, 1) - get(R, 0, 1) * get(R, 1, 0);
    expect(det).toBeCloseTo(1, 12);
  });
});

describe("buildAFromFactors", () => {
  it("reconstructs A = U diag(σ) Vᵀ", () => {
    const { A, U, V, sigma } = buildAFromFactors({
      s1: 2.5,
      s2: 0.4,
      thVDeg: 30,
      thUDeg: -15,
    });
    const svd = classicalSvd(A, 2);
    expect(svd.sigma[0]).toBeCloseTo(sigma[0], 8);
    expect(svd.sigma[1]).toBeCloseTo(sigma[1], 8);
    // Columns of U,V are unit length
    expect(vecLen(get(U, 0, 0), get(U, 1, 0))).toBeCloseTo(1, 12);
    expect(vecLen(get(V, 0, 0), get(V, 1, 0))).toBeCloseTo(1, 12);
  });

  it("matches classical SVD reconstruction in Frobenius norm", () => {
    const { A } = buildAFromFactors({
      s1: 1.8,
      s2: 0.9,
      thVDeg: 40,
      thUDeg: 10,
    });
    const f = frameFromMatrix(A);
    expect(frobeniusSq(sub(A, f.A))).toBe(0);
    expect(f.sigma[0]).toBeGreaterThanOrEqual(f.sigma[1]);
  });
});

describe("frameFromMatrix / movieStages", () => {
  it("U Σ Vᵀ stages agree with Ax on sample vectors", () => {
    const A = fromNested([
      [1.2, 0.9],
      [0.4, 1.5],
    ]);
    const f = frameFromMatrix(A);
    for (const ang of [0, 0.4, 1.1, 2.0, Math.PI]) {
      const x = Math.cos(ang);
      const y = Math.sin(ang);
      const { afterU, direct } = movieStages(f, x, y);
      expect(afterU[0]).toBeCloseTo(direct[0], 9);
      expect(afterU[1]).toBeCloseTo(direct[1], 9);
    }
  });

  it("Avⱼ = σⱼ uⱼ for singular vectors", () => {
    const f = frameFromMatrix(
      fromNested([
        [0.8, -0.5],
        [0.3, 1.1],
      ]),
    );
    for (let j = 0; j < 2; j++) {
      const [ax, ay] = applyMat2(f.A, get(f.V, 0, j), get(f.V, 1, j));
      expect(ax).toBeCloseTo(get(f.U, 0, j) * f.sigma[j], 8);
      expect(ay).toBeCloseTo(get(f.U, 1, j) * f.sigma[j], 8);
    }
  });
});

describe("sampleStretchRange", () => {
  it("matches singular values for a full-rank matrix", () => {
    const A = fromNested([
      [1.2, 0.9],
      [0.4, 1.5],
    ]);
    const { sigma } = classicalSvd(A, 2);
    const { max, min } = sampleStretchRange(A, 720);
    expect(max).toBeCloseTo(sigma[0]!, 2);
    expect(min).toBeCloseTo(sigma[1]!, 2);
  });

  it("collapses min stretch near 0 for a rank-1 matrix", () => {
    const { A } = buildAFromFactors({
      s1: 2,
      s2: 0,
      thVDeg: 20,
      thUDeg: 50,
    });
    const { min, max } = sampleStretchRange(A, 720);
    expect(max).toBeCloseTo(2, 2);
    expect(min).toBeLessThan(1e-6);
  });
});

describe("TILTED_EXAMPLE_A", () => {
  it("has singular values different from the diagonal entries", () => {
    const f = frameFromMatrix(TILTED_EXAMPLE_A);
    expect(f.sigma[0]).not.toBeCloseTo(1.5, 1);
    expect(f.sigma[1]).not.toBeCloseTo(1.2, 1);
    expect(f.sigma[0]).toBeGreaterThan(f.sigma[1]);
    // Off-diagonals are nonzero — genuinely tilted
    expect(Math.abs(get(TILTED_EXAMPLE_A, 0, 1))).toBeGreaterThan(0.1);
    expect(Math.abs(get(TILTED_EXAMPLE_A, 1, 0))).toBeGreaterThan(0.1);
  });
});

describe("frameFromFactors", () => {
  it("preserves the scrubbed singular values exactly", () => {
    const f = frameFromFactors({
      s1: 2.2,
      s2: 0.5,
      thVDeg: -25,
      thUDeg: 40,
    });
    expect(f.sigma[0]).toBe(2.2);
    expect(f.sigma[1]).toBe(0.5);
    const { max, min } = sampleStretchRange(f.A, 720);
    expect(max).toBeCloseTo(2.2, 2);
    expect(min).toBeCloseTo(0.5, 2);
  });
});
