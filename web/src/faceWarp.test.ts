import { describe, expect, it } from "vitest";
import { parseAsf, toPixels } from "./immAsf";
import {
  procrustesAlign,
  triangulate,
  fitShapeToCanvas,
  applyCanvasFit,
} from "./faceWarp";
import { classicalSvd } from "./classicalSvd";
import { mat, set, get } from "./matrix";

describe("parseAsf", () => {
  it("reads relative landmark lines after the count", () => {
    const text = `#\n2\n#\n0 4 0.25 0.50 0 0 1\n0 4 0.75 0.50 1 0 0\n#\nhost.jpg\n`;
    const pts = parseAsf(text);
    expect(pts).toHaveLength(2);
    expect(pts[0]).toEqual({ x: 0.25, y: 0.5 });
    expect(toPixels(pts, 100, 200)[1]).toEqual({ x: 75, y: 100 });
  });
});

describe("procrustes + delaunay", () => {
  it("aligns a translated/scaled copy toward the target", () => {
    const target = [
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 1, y: 2 },
    ];
    const from = target.map((p) => ({ x: 10 + 3 * p.x, y: 5 + 3 * p.y }));
    const aligned = procrustesAlign(from, target);
    const err =
      aligned.reduce((s, p, i) => s + (p.x - target[i]!.x) ** 2 + (p.y - target[i]!.y) ** 2, 0) /
      aligned.length;
    expect(err).toBeLessThan(1e-6);
  });

  it("triangulates mean shape with delaunator", () => {
    const s = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0.5, y: 1 },
      { x: 0.5, y: 0.3 },
    ];
    const tri = triangulate(s);
    expect(tri.length % 3).toBe(0);
    expect(tri.length).toBeGreaterThanOrEqual(3);
  });

  it("shares one canvas fit across shapes", () => {
    const a = [
      { x: 10, y: 10 },
      { x: 20, y: 10 },
      { x: 15, y: 25 },
    ];
    const fit = fitShapeToCanvas(a, 64);
    const on = applyCanvasFit(a, fit);
    expect(Math.min(...on.map((p) => p.x))).toBeGreaterThanOrEqual(4 - 1e-6);
    expect(Math.max(...on.map((p) => p.x))).toBeLessThanOrEqual(60 + 1e-6);
  });
});

describe("appearance SVD codes", () => {
  it("keeps singular values descending on a synthetic stack", () => {
    const nPix = 16;
    const N = 8;
    const X = mat(nPix, N);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < nPix; i++) {
        set(X, i, j, Math.sin(i / 3) + 0.3 * j * Math.cos(i / 2));
      }
    }
    const mean = new Float64Array(nPix);
    for (let i = 0; i < nPix; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += get(X, i, j);
      mean[i] = s / N;
    }
    const C = mat(nPix, N);
    for (let i = 0; i < nPix; i++) {
      for (let j = 0; j < N; j++) set(C, i, j, get(X, i, j) - mean[i]!);
    }
    const { U, sigma } = classicalSvd(C, 2);
    expect(sigma[0]!).toBeGreaterThan(sigma[1]!);
    expect(U.cols).toBe(2);
  });
});
