import { describe, expect, it } from "vitest";
import {
  applyLeft,
  applyMat2,
  applyRight,
  columnOf,
  householderAimColumn,
  householderAimToE1,
  householderFromNormal,
  reflectAcrossNormal,
  rightGivensZeroSuperdiag,
} from "./householder2d";
import { frobeniusSq, fromNested, get, matmul, transpose } from "./matrix";

describe("householderFromNormal", () => {
  it("is orthogonal and an involution", () => {
    const H = householderFromNormal(1, 2);
    const HtH = matmul(transpose(H), H);
    expect(get(HtH, 0, 0)).toBeCloseTo(1, 10);
    expect(get(HtH, 1, 1)).toBeCloseTo(1, 10);
    expect(get(HtH, 0, 1)).toBeCloseTo(0, 10);
    const HH = matmul(H, H);
    expect(get(HH, 0, 0)).toBeCloseTo(1, 10);
    expect(get(HH, 1, 1)).toBeCloseTo(1, 10);
    expect(get(HH, 0, 1)).toBeCloseTo(0, 10);
  });

  it("matches reflectAcrossNormal", () => {
    const nx = 0.3;
    const ny = 0.8;
    const H = householderFromNormal(nx, ny);
    const [rx, ry] = reflectAcrossNormal(1.2, -0.4, nx, ny);
    const [mx, my] = applyMat2(H, 1.2, -0.4);
    expect(mx).toBeCloseTo(rx, 10);
    expect(my).toBeCloseTo(ry, 10);
  });
});

describe("householderAimToE1", () => {
  it("sends a to the x-axis with the same length", () => {
    const ax = 1.2;
    const ay = 0.7;
    const { H, normA, targetX } = householderAimToE1(ax, ay);
    const [ox, oy] = applyMat2(H, ax, ay);
    expect(oy).toBeCloseTo(0, 10);
    expect(Math.abs(ox)).toBeCloseTo(normA, 10);
    expect(Math.abs(ox)).toBeCloseTo(Math.abs(targetX), 10);
  });

  it("preserves lengths of other vectors", () => {
    const { H } = householderAimToE1(0.5, 1.1);
    const [ox, oy] = applyMat2(H, 0.3, -0.8);
    expect(Math.hypot(ox, oy)).toBeCloseTo(Math.hypot(0.3, -0.8), 10);
  });
});

describe("householderAimColumn", () => {
  it("zeros below the first entry of a column", () => {
    const a = [1.0, 0.4, -0.6, 0.2, 0.3];
    const H = householderAimColumn(a);
    const A = fromNested(a.map((v) => [v]));
    const HA = applyLeft(H, A);
    expect(Math.abs(get(HA, 0, 0))).toBeCloseTo(Math.hypot(...a), 8);
    for (let i = 1; i < a.length; i++) {
      expect(get(HA, i, 0)).toBeCloseTo(0, 8);
    }
  });

  it("left-multiplies a square matrix and zeros subdiagonal of col 0", () => {
    const A = fromNested([
      [1.2, 0.3, -0.1, 0.4, 0.2],
      [0.5, 1.0, 0.2, -0.3, 0.1],
      [-0.4, 0.2, 0.8, 0.1, -0.2],
      [0.3, -0.1, 0.2, 1.1, 0.0],
      [0.2, 0.4, -0.3, 0.2, 0.9],
    ]);
    const H = householderAimColumn(columnOf(A, 0));
    const HA = applyLeft(H, A);
    for (let i = 1; i < 5; i++) {
      expect(get(HA, i, 0)).toBeCloseTo(0, 8);
    }
    expect(Math.sqrt(frobeniusSq(HA))).toBeCloseTo(Math.sqrt(frobeniusSq(A)), 8);
  });
});

describe("rightGivensZeroSuperdiag", () => {
  it("zeros the (0,1) entry of an upper-triangular 2×2", () => {
    const A = fromNested([
      [2.0, 1.5],
      [0.0, 0.8],
    ]);
    const G = rightGivensZeroSuperdiag(A);
    const R = applyRight(A, G);
    expect(get(R, 0, 1)).toBeCloseTo(0, 10);
  });
});
