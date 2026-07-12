/** Appearance + shape SVD models for aligned IMM faces; bottleneck noise sampling. */

import { classicalSvd } from "./classicalSvd";
import { type Matrix, mat, get, set } from "./matrix";
import {
  type Shape,
  type RgbaBitmap,
  meanShape,
  procrustesAlign,
  flattenShape,
  unflattenShape,
  triangulate,
  warpPiecewiseAffine,
  fitShapeToCanvas,
  applyCanvasFit,
} from "./faceWarp";
import { parseAsf, toPixels } from "./immAsf";
import { fetchImmPack } from "./immPack";

export const FACE_SIZE = 64;

export type FaceExample = {
  id: string;
  /** Original grayscale crop thumbnail (bbox), for filmstrip. */
  thumb: Float64Array;
  /** Shape-normalized appearance (FACE_SIZE²), values in [0,1]. */
  appearance: Float64Array;
  /** Landmarks in FACE_SIZE canvas coords (shared mean-shape frame). */
  shape: Shape;
};

export type FaceModel = {
  size: number;
  meanAppearance: Float64Array;
  /** Columns = appearance principal directions (pixels × k). */
  appearanceU: Matrix;
  appearanceSigma: number[];
  /** Codes for each training example (N × k). */
  appearanceCodes: Matrix;
  meanShape: Shape;
  shapeU: Matrix;
  shapeSigma: number[];
  shapeCodes: Matrix;
  triangles: Uint32Array;
  examples: FaceExample[];
};

export type ImmManifest = {
  citation: string;
  files: string[];
};

export type LoadedImmFace = {
  id: string;
  img: RgbaBitmap;
  shapePx: Shape;
};

/** Convert an RGBA bitmap to grayscale in-place (R=G=B=luma). */
export function rgbaToGrayInPlace(img: RgbaBitmap): void {
  const data = img.data;
  for (let i = 0; i < data.length; i += 4) {
    const y = 0.299 * data[i]! + 0.587 * data[i + 1]! + 0.114 * data[i + 2]!;
    data[i] = data[i + 1] = data[i + 2] = y;
  }
}

let grayScratch: HTMLCanvasElement | null = null;

function bitmapToGrayData(bitmap: ImageBitmap): RgbaBitmap {
  const c = grayScratch ?? (grayScratch = document.createElement("canvas"));
  c.width = bitmap.width;
  c.height = bitmap.height;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const data = ctx.getImageData(0, 0, c.width, c.height);
  rgbaToGrayInPlace(data);
  return data;
}

/** Run `fn` over `items` with at most `concurrency` in flight; preserve order. */
async function mapPool<T, R>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
  onProgress?: (done: number, total: number) => void,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  let done = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i]!, i);
      done++;
      onProgress?.(done, items.length);
    }
  });
  await Promise.all(workers);
  return results;
}

async function loadOneImm(
  baseUrl: string,
  id: string,
): Promise<LoadedImmFace> {
  const [imgRes, asfRes] = await Promise.all([
    fetch(`${baseUrl}/${id}.jpg`),
    fetch(`${baseUrl}/${id}.asf`),
  ]);
  if (!imgRes.ok) throw new Error(`JPG ${id}: ${imgRes.status}`);
  if (!asfRes.ok) throw new Error(`ASF ${id}: ${asfRes.status}`);
  const [blob, asfText] = await Promise.all([imgRes.blob(), asfRes.text()]);
  const bitmap = await createImageBitmap(blob);
  const img = bitmapToGrayData(bitmap);
  const shapePx = toPixels(parseAsf(asfText), img.width, img.height);
  return { id, img, shapePx };
}

export function bboxThumb(src: RgbaBitmap, shape: Shape, size: number): Float64Array {
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
  const pad = 0.15 * Math.max(maxX - minX, maxY - minY);
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(src.width, maxX + pad);
  maxY = Math.min(src.height, maxY + pad);
  const out = new Float64Array(size * size);
  for (let yi = 0; yi < size; yi++) {
    for (let xi = 0; xi < size; xi++) {
      const sx = minX + ((xi + 0.5) / size) * (maxX - minX);
      const sy = minY + ((yi + 0.5) / size) * (maxY - minY);
      const x0 = Math.min(src.width - 1, Math.max(0, Math.floor(sx)));
      const y0 = Math.min(src.height - 1, Math.max(0, Math.floor(sy)));
      out[yi * size + xi] = src.data[(y0 * src.width + x0) * 4]! / 255;
    }
  }
  return out;
}

/**
 * Shared align + warp pipeline (browser or Node bake).
 * Same code path either way — bake just supplies `loaded` from disk.
 */
export async function buildExamplesFromLoaded(
  loaded: LoadedImmFace[],
  size = FACE_SIZE,
  onProgress?: (msg: string) => void,
): Promise<FaceExample[]> {
  const n = loaded.length;
  onProgress?.(`Aligning ${n} shapes…`);
  let mean = meanShape(loaded.map((e) => e.shapePx));
  for (let iter = 0; iter < 4; iter++) {
    mean = meanShape(loaded.map((e) => procrustesAlign(e.shapePx, mean)));
  }

  const fit = fitShapeToCanvas(mean, size);
  const meanOnCanvas = applyCanvasFit(mean, fit);
  const triangles = triangulate(meanOnCanvas);
  const examples: FaceExample[] = [];

  for (let i = 0; i < loaded.length; i++) {
    const { id, img, shapePx } = loaded[i]!;
    onProgress?.(`Warping ${i + 1}/${n}…`);
    const aligned = procrustesAlign(shapePx, mean);
    const shape = applyCanvasFit(aligned, fit);
    const appearance = warpPiecewiseAffine(img, shapePx, meanOnCanvas, triangles, size);
    const thumb = bboxThumb(img, shapePx, size);
    examples.push({ id, thumb, appearance, shape });
    if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
  }

  return examples;
}

/** Concurrent face fetches; browsers multiplex well under HTTP/2. */
const LOAD_CONCURRENCY = 16;

/** Live path: fetch raw IMM jpg+asf and warp in-process (also what the bake script mirrors). */
export async function loadImmExamples(
  baseUrl: string,
  files: string[],
  size = FACE_SIZE,
  onProgress?: (msg: string) => void,
): Promise<{ examples: FaceExample[] }> {
  const n = files.length;
  onProgress?.(`Loading 0/${n}…`);
  const loaded = await mapPool(
    files,
    LOAD_CONCURRENCY,
    (id) => loadOneImm(baseUrl, id),
    (done, total) => onProgress?.(`Loading ${done}/${total}…`),
  );
  const examples = await buildExamplesFromLoaded(loaded, size, onProgress);
  return { examples };
}

/** Fast path: one pre-warped pack produced by `npm run gen:imm-pack`. */
export async function loadImmExamplesPack(
  url: string,
  onProgress?: (msg: string) => void,
): Promise<{ examples: FaceExample[] }> {
  const { examples } = await fetchImmPack(url, onProgress);
  return { examples: examples as FaceExample[] };
}

function appearanceMatrix(examples: FaceExample[]): Matrix {
  const nPix = FACE_SIZE * FACE_SIZE;
  const X = mat(nPix, examples.length);
  for (let j = 0; j < examples.length; j++) {
    const a = examples[j]!.appearance;
    for (let i = 0; i < nPix; i++) set(X, i, j, a[i]!);
  }
  return X;
}

function shapeMatrix(examples: FaceExample[]): Matrix {
  const dim = examples[0]!.shape.length * 2;
  const X = mat(dim, examples.length);
  for (let j = 0; j < examples.length; j++) {
    const v = flattenShape(examples[j]!.shape);
    for (let i = 0; i < dim; i++) set(X, i, j, v[i]!);
  }
  return X;
}

function centerColumns(X: Matrix): { centered: Matrix; mean: Float64Array } {
  const mean = new Float64Array(X.rows);
  for (let i = 0; i < X.rows; i++) {
    let s = 0;
    for (let j = 0; j < X.cols; j++) s += get(X, i, j);
    mean[i] = s / X.cols;
  }
  const centered = mat(X.rows, X.cols);
  for (let i = 0; i < X.rows; i++) {
    for (let j = 0; j < X.cols; j++) set(centered, i, j, get(X, i, j) - mean[i]!);
  }
  return { centered, mean };
}

function codesFromU(Uc: Matrix, centered: Matrix): Matrix {
  // codes = U^T X  (k × N)
  const k = Uc.cols;
  const N = centered.cols;
  const C = mat(N, k);
  for (let j = 0; j < N; j++) {
    for (let ell = 0; ell < k; ell++) {
      let s = 0;
      for (let i = 0; i < centered.rows; i++) {
        s += get(Uc, i, ell) * get(centered, i, j);
      }
      set(C, j, ell, s);
    }
  }
  return C;
}

export function buildFaceModel(examples: FaceExample[], rank: number): FaceModel {
  const k = Math.max(1, Math.min(rank, examples.length - 1, FACE_SIZE * FACE_SIZE));
  const meanSh = meanShape(examples.map((e) => e.shape));
  const triangles = triangulate(meanSh);

  const { centered: appC, mean: meanAppearance } = centerColumns(appearanceMatrix(examples));
  const appSvd = classicalSvd(appC, k);
  const appearanceCodes = codesFromU(appSvd.U, appC);

  const { centered: shC, mean: meanShapeFlat } = centerColumns(shapeMatrix(examples));
  const shapeRank = Math.max(1, Math.min(k, examples.length - 1, meanShapeFlat.length));
  const shSvd = classicalSvd(shC, shapeRank);
  const shapeCodes = codesFromU(shSvd.U, shC);

  return {
    size: FACE_SIZE,
    meanAppearance,
    appearanceU: appSvd.U,
    appearanceSigma: appSvd.sigma,
    appearanceCodes,
    meanShape: unflattenShape(meanShapeFlat),
    shapeU: shSvd.U,
    shapeSigma: shSvd.sigma,
    shapeCodes,
    triangles,
    examples,
  };
}

function randn(): number {
  const u = Math.max(1e-12, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Decode appearance vector (length size²) from code. */
export function decodeAppearance(model: FaceModel, code: ArrayLike<number>, k: number): Float64Array {
  const kk = Math.min(k, model.appearanceU.cols, code.length);
  const out = new Float64Array(model.meanAppearance);
  for (let ell = 0; ell < kk; ell++) {
    const c = code[ell]!;
    for (let i = 0; i < out.length; i++) {
      out[i]! += get(model.appearanceU, i, ell) * c;
    }
  }
  return out;
}

/** Bottleneck coordinates for training example `idx` (length = full model rank). */
export function getAppearanceCode(model: FaceModel, idx: number): Float64Array {
  const k = model.appearanceCodes.cols;
  const code = new Float64Array(k);
  for (let ell = 0; ell < k; ell++) code[ell] = get(model.appearanceCodes, idx, ell);
  return code;
}

/** Mean-squared error between two grayscale maps in [0,1]. */
export function mseGray(a: Float64Array, b: Float64Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!;
    s += d * d;
  }
  return s / Math.max(n, 1);
}

export function decodeShape(model: FaceModel, code: ArrayLike<number>, k: number): Shape {
  const kk = Math.min(k, model.shapeU.cols, code.length);
  const flat = flattenShape(model.meanShape);
  for (let ell = 0; ell < kk; ell++) {
    const c = code[ell]!;
    for (let i = 0; i < flat.length; i++) {
      flat[i]! += get(model.shapeU, i, ell) * c;
    }
  }
  return unflattenShape(flat);
}

/**
 * Sample: pick a random training code, add τ·N(0,1) in bottleneck coords
 * (scaled by singular values so noise is in a meaningful units), decode.
 * Returns appearance in mean-shape frame (sharp correspondence).
 */
export function sampleNoisyAppearance(
  model: FaceModel,
  k: number,
  tau: number,
): Float64Array {
  const idx = Math.floor(Math.random() * model.appearanceCodes.rows);
  const kk = Math.min(k, model.appearanceCodes.cols);
  const code = new Float64Array(kk);
  for (let ell = 0; ell < kk; ell++) {
    const base = get(model.appearanceCodes, idx, ell);
    const scale = model.appearanceSigma[ell] ?? 1;
    code[ell] = base + tau * scale * randn();
  }
  return decodeAppearance(model, code, kk);
}

/** Pixel-space SVD foil: codes on bbox thumbs with no warp correspondence. */
export function buildPixelFoilModel(examples: FaceExample[], rank: number): {
  mean: Float64Array;
  U: Matrix;
  sigma: number[];
  codes: Matrix;
} {
  const nPix = FACE_SIZE * FACE_SIZE;
  const X = mat(nPix, examples.length);
  for (let j = 0; j < examples.length; j++) {
    const a = examples[j]!.thumb;
    for (let i = 0; i < nPix; i++) set(X, i, j, a[i]!);
  }
  const { centered, mean } = centerColumns(X);
  const k = Math.max(1, Math.min(rank, examples.length - 1));
  const svd = classicalSvd(centered, k);
  const codes = codesFromU(svd.U, centered);
  return { mean, U: svd.U, sigma: svd.sigma, codes };
}

export function samplePixelFoil(
  foil: { mean: Float64Array; U: Matrix; sigma: number[]; codes: Matrix },
  k: number,
  tau: number,
): Float64Array {
  const idx = Math.floor(Math.random() * foil.codes.rows);
  const kk = Math.min(k, foil.codes.cols);
  const out = new Float64Array(foil.mean);
  for (let ell = 0; ell < kk; ell++) {
    const base = get(foil.codes, idx, ell);
    const scale = foil.sigma[ell] ?? 1;
    const c = base + tau * scale * randn();
    for (let i = 0; i < out.length; i++) out[i]! += get(foil.U, i, ell) * c;
  }
  return out;
}

export function grayToImageData(gray: Float64Array, size: number): ImageData {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const v = Math.min(255, Math.max(0, Math.round(gray[i]! * 255)));
    data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = v;
    data[4 * i + 3] = 255;
  }
  return new ImageData(data, size, size);
}

export function drawGray(
  canvas: HTMLCanvasElement,
  gray: Float64Array,
  size: number,
): void {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(grayToImageData(gray, size), 0, 0);
}
