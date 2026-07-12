/** Procrustes alignment and piecewise-affine warps via delaunator. */

import Delaunator from "delaunator";
import type { Point } from "./immAsf";

export type Shape = Point[];

export function meanShape(shapes: Shape[]): Shape {
  const n = shapes[0]?.length ?? 0;
  const out: Shape = Array.from({ length: n }, () => ({ x: 0, y: 0 }));
  for (const s of shapes) {
    for (let i = 0; i < n; i++) {
      out[i].x += s[i].x;
      out[i].y += s[i].y;
    }
  }
  const inv = 1 / Math.max(shapes.length, 1);
  for (const p of out) {
    p.x *= inv;
    p.y *= inv;
  }
  return out;
}

function centroid(s: Shape): Point {
  let x = 0;
  let y = 0;
  for (const p of s) {
    x += p.x;
    y += p.y;
  }
  const inv = 1 / Math.max(s.length, 1);
  return { x: x * inv, y: y * inv };
}

function centered(s: Shape): Shape {
  const c = centroid(s);
  return s.map((p) => ({ x: p.x - c.x, y: p.y - c.y }));
}

/** Similarity Procrustes: map `from` onto `to` (same landmark count). */
export function procrustesAlign(from: Shape, to: Shape): Shape {
  const a = centered(from);
  const b = centered(to);
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += a[i].x * b[i].y - a[i].y * b[i].x;
    den += a[i].x * b[i].x + a[i].y * b[i].y;
  }
  const theta = Math.atan2(num, den);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  let normA = 0;
  let cross = 0;
  for (let i = 0; i < a.length; i++) {
    const rx = cos * a[i].x - sin * a[i].y;
    const ry = sin * a[i].x + cos * a[i].y;
    a[i] = { x: rx, y: ry };
    normA += rx * rx + ry * ry;
    cross += rx * b[i].x + ry * b[i].y;
  }
  const scale = cross / Math.max(normA, 1e-12);
  const cTo = centroid(to);
  return a.map((p) => ({
    x: scale * p.x + cTo.x,
    y: scale * p.y + cTo.y,
  }));
}

export function flattenShape(s: Shape): Float64Array {
  const out = new Float64Array(s.length * 2);
  for (let i = 0; i < s.length; i++) {
    out[2 * i] = s[i].x;
    out[2 * i + 1] = s[i].y;
  }
  return out;
}

export function unflattenShape(v: ArrayLike<number>): Shape {
  const n = Math.floor(v.length / 2);
  const out: Shape = [];
  for (let i = 0; i < n; i++) out.push({ x: v[2 * i], y: v[2 * i + 1] });
  return out;
}

export function triangulate(shape: Shape): Uint32Array {
  const coords = new Float64Array(shape.length * 2);
  for (let i = 0; i < shape.length; i++) {
    coords[2 * i] = shape[i].x;
    coords[2 * i + 1] = shape[i].y;
  }
  return new Delaunator(coords).triangles;
}

function barycentric(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
): { u: number; v: number; w: number } | null {
  const v0x = bx - ax;
  const v0y = by - ay;
  const v1x = cx - ax;
  const v1y = cy - ay;
  const v2x = px - ax;
  const v2y = py - ay;
  const den = v0x * v1y - v1x * v0y;
  if (Math.abs(den) < 1e-12) return null;
  const v = (v2x * v1y - v1x * v2y) / den;
  const w = (v0x * v2y - v2x * v0y) / den;
  const u = 1 - v - w;
  return { u, v, w };
}

function sampleBilinear(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  x: number,
  y: number,
): number {
  if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) {
    const xi = Math.min(w - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(h - 1, Math.max(0, Math.round(y)));
    return data[(yi * w + xi) * 4]!;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const i00 = (y0 * w + x0) * 4;
  const i10 = (y0 * w + x0 + 1) * 4;
  const i01 = ((y0 + 1) * w + x0) * 4;
  const i11 = ((y0 + 1) * w + x0 + 1) * 4;
  const a = data[i00]!;
  const b = data[i10]!;
  const c = data[i01]!;
  const d = data[i11]!;
  return (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d;
}

/**
 * Inverse piecewise-affine warp: for each dest pixel, find its triangle in
 * `dstShape`, map to `srcShape`, and sample grayscale from `src` ImageData.
 */
export function warpPiecewiseAffine(
  src: ImageData,
  srcShape: Shape,
  dstShape: Shape,
  triangles: Uint32Array,
  outSize: number,
): Float64Array {
  const out = new Float64Array(outSize * outSize);
  const sw = src.width;
  const sh = src.height;
  const data = src.data;

  // Per-triangle axis-aligned bounds for a cheap reject.
  const nTri = triangles.length / 3;
  const tb = new Float64Array(nTri * 4);
  for (let t = 0, ti = 0; t < triangles.length; t += 3, ti++) {
    const d0 = dstShape[triangles[t]!]!;
    const d1 = dstShape[triangles[t + 1]!]!;
    const d2 = dstShape[triangles[t + 2]!]!;
    tb[ti * 4] = Math.min(d0.x, d1.x, d2.x);
    tb[ti * 4 + 1] = Math.min(d0.y, d1.y, d2.y);
    tb[ti * 4 + 2] = Math.max(d0.x, d1.x, d2.x);
    tb[ti * 4 + 3] = Math.max(d0.y, d1.y, d2.y);
  }

  for (let yi = 0; yi < outSize; yi++) {
    for (let xi = 0; xi < outSize; xi++) {
      const px = xi + 0.5;
      const py = yi + 0.5;
      let found = false;
      for (let ti = 0, t = 0; t < triangles.length; t += 3, ti++) {
        if (
          px < tb[ti * 4]! - 0.5 ||
          py < tb[ti * 4 + 1]! - 0.5 ||
          px > tb[ti * 4 + 2]! + 0.5 ||
          py > tb[ti * 4 + 3]! + 0.5
        ) {
          continue;
        }
        const i0 = triangles[t]!;
        const i1 = triangles[t + 1]!;
        const i2 = triangles[t + 2]!;
        const d0 = dstShape[i0]!;
        const d1 = dstShape[i1]!;
        const d2 = dstShape[i2]!;
        const bc = barycentric(px, py, d0.x, d0.y, d1.x, d1.y, d2.x, d2.y);
        if (!bc || bc.u < -1e-4 || bc.v < -1e-4 || bc.w < -1e-4) continue;
        const s0 = srcShape[i0]!;
        const s1 = srcShape[i1]!;
        const s2 = srcShape[i2]!;
        const sx = bc.u * s0.x + bc.v * s1.x + bc.w * s2.x;
        const sy = bc.u * s0.y + bc.v * s1.y + bc.w * s2.y;
        out[yi * outSize + xi] = sampleBilinear(data, sw, sh, sx, sy) / 255;
        found = true;
        break;
      }
      if (!found) out[yi * outSize + xi] = 0;
    }
  }
  return out;
}

export type CanvasFit = {
  minX: number;
  minY: number;
  scale: number;
  ox: number;
  oy: number;
};

/** Fit params so `shape` lands in [pad, size-pad]². Apply with `applyCanvasFit`. */
export function fitShapeToCanvas(shape: Shape, size: number, pad = 4): CanvasFit {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of shape) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const bw = Math.max(maxX - minX, 1e-6);
  const bh = Math.max(maxY - minY, 1e-6);
  const inner = size - 2 * pad;
  const scale = Math.min(inner / bw, inner / bh);
  const ox = pad + (inner - bw * scale) / 2;
  const oy = pad + (inner - bh * scale) / 2;
  return { minX, minY, scale, ox, oy };
}

export function applyCanvasFit(shape: Shape, fit: CanvasFit): Shape {
  return shape.map((p) => ({
    x: (p.x - fit.minX) * fit.scale + fit.ox,
    y: (p.y - fit.minY) * fit.scale + fit.oy,
  }));
}

export function normalizeShapeToCanvas(shape: Shape, size: number, pad = 4): Shape {
  return applyCanvasFit(shape, fitShapeToCanvas(shape, size, pad));
}
