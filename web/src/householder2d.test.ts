import { describe, expect, it } from "vitest";
import {
  applyLeft,
  applyMat2,
  applyRight,
  columnOf,
  decomposeAlongNormal,
  householderAimColumn,
  householderAimToE1,
  householderFromNormal,
  normalFromLineAngle,
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

describe("normalFromLineAngle (mirror slider)", () => {
  it("is always unit length across the slider range", () => {
    for (let ang = -90; ang <= 90; ang += 1) {
      const [nx, ny] = normalFromLineAngle(ang);
      expect(Math.hypot(nx, ny)).toBeCloseTo(1, 12);
    }
  });

  it("is orthogonal to the mirror direction", () => {
    for (const ang of [-90, -45, 0, 35, 45, 90]) {
      const th = (ang * Math.PI) / 180;
      const [nx, ny] = normalFromLineAngle(ang);
      expect(nx * Math.cos(th) + ny * Math.sin(th)).toBeCloseTo(0, 12);
    }
  });

  it("builds H that reflects the probe for every mirror angle", () => {
    const probe: [number, number] = [1.1, 0.55];
    for (let ang = -90; ang <= 90; ang += 5) {
      const [nx, ny] = normalFromLineAngle(ang);
      const H = householderFromNormal(nx, ny);
      const [hx, hy] = applyMat2(H, probe[0], probe[1]);
      const [rx, ry] = reflectAcrossNormal(probe[0], probe[1], nx, ny);
      expect(hx).toBeCloseTo(rx, 10);
      expect(hy).toBeCloseTo(ry, 10);
      expect(Math.hypot(hx, hy)).toBeCloseTo(Math.hypot(probe[0], probe[1]), 10);

      const { parallel, normal } = decomposeAlongNormal(
        probe[0],
        probe[1],
        nx,
        ny,
      );
      expect(hx).toBeCloseTo(parallel[0] - normal[0], 10);
      expect(hy).toBeCloseTo(parallel[1] - normal[1], 10);
      expect(hx).toBeCloseTo(probe[0] - 2 * normal[0], 10);
      expect(hy).toBeCloseTo(probe[1] - 2 * normal[1], 10);
    }
  });
});

describe("decomposeAlongNormal", () => {
  it("reconstructs x = parallel + normal", () => {
    const { parallel, normal } = decomposeAlongNormal(1.1, 0.55, -0.5, 0.8);
    expect(parallel[0] + normal[0]).toBeCloseTo(1.1, 12);
    expect(parallel[1] + normal[1]).toBeCloseTo(0.55, 12);
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
