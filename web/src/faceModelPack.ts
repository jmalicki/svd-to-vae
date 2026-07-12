/** Binary pack of precomputed face / foil SVD models (build-time or in-browser). */

import { type Matrix, mat } from "./matrix";
import type { FaceExample, FaceModel } from "./faceModel";
import { unflattenShape, type Shape } from "./faceWarp";

export type PixelFoilModel = {
  mean: Float64Array;
  U: Matrix;
  sigma: number[];
  codes: Matrix;
};

export const FACE_MODEL_PACK_MAGIC = 0x444d4346; // "FCMD" little-endian
export const FACE_MODEL_PACK_VERSION = 1;

function writeF32(view: DataView, byteOffset: number, src: ArrayLike<number>): number {
  for (let i = 0; i < src.length; i++) view.setFloat32(byteOffset + 4 * i, src[i]!, true);
  return byteOffset + 4 * src.length;
}

function readF32(view: DataView, byteOffset: number, n: number): { values: Float64Array; next: number } {
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = view.getFloat32(byteOffset + 4 * i, true);
  return { values, next: byteOffset + 4 * n };
}

function writeMatrix(view: DataView, byteOffset: number, m: Matrix): number {
  return writeF32(view, byteOffset, m.data);
}

function readMatrix(
  view: DataView,
  byteOffset: number,
  rows: number,
  cols: number,
): { matrix: Matrix; next: number } {
  const { values, next } = readF32(view, byteOffset, rows * cols);
  const matrix = mat(rows, cols);
  matrix.data.set(values);
  return { matrix, next };
}

/**
 * Pack appearance + shape SVD and pixel-foil SVD (no examples — those stay in examples.bin).
 * Float32 payload; runtime promotes to Float64 Matrix / arrays.
 */
export function encodeFaceModelPack(model: FaceModel, foil: PixelFoilModel): ArrayBuffer {
  const n = model.appearanceCodes.rows;
  const kApp = model.appearanceU.cols;
  const kShape = model.shapeU.cols;
  const kFoil = foil.U.cols;
  const size = model.size;
  const nLandmarks = model.meanShape.length;
  const nTriIdx = model.triangles.length;
  const shapeDim = nLandmarks * 2;
  const nPix = size * size;

  // header: magic, version, n, kApp, kShape, kFoil, size, nLandmarks, nTriIdx (9 × u32)
  const header = 36;
  const f32Count =
    nPix +
    nPix * kApp +
    kApp +
    n * kApp +
    shapeDim +
    shapeDim * kShape +
    kShape +
    n * kShape +
    nPix +
    nPix * kFoil +
    kFoil +
    n * kFoil;
  const buf = new ArrayBuffer(header + 4 * f32Count + 4 * nTriIdx);
  const view = new DataView(buf);
  view.setUint32(0, FACE_MODEL_PACK_MAGIC, true);
  view.setUint32(4, FACE_MODEL_PACK_VERSION, true);
  view.setUint32(8, n, true);
  view.setUint32(12, kApp, true);
  view.setUint32(16, kShape, true);
  view.setUint32(20, kFoil, true);
  view.setUint32(24, size, true);
  view.setUint32(28, nLandmarks, true);
  view.setUint32(32, nTriIdx, true);

  let off = header;
  off = writeF32(view, off, model.meanAppearance);
  off = writeMatrix(view, off, model.appearanceU);
  off = writeF32(view, off, model.appearanceSigma);
  off = writeMatrix(view, off, model.appearanceCodes);

  const meanFlat = new Float64Array(shapeDim);
  for (let i = 0; i < nLandmarks; i++) {
    meanFlat[2 * i] = model.meanShape[i]!.x;
    meanFlat[2 * i + 1] = model.meanShape[i]!.y;
  }
  off = writeF32(view, off, meanFlat);
  off = writeMatrix(view, off, model.shapeU);
  off = writeF32(view, off, model.shapeSigma);
  off = writeMatrix(view, off, model.shapeCodes);

  for (let i = 0; i < nTriIdx; i++) {
    view.setUint32(off + 4 * i, model.triangles[i]!, true);
  }
  off += 4 * nTriIdx;

  off = writeF32(view, off, foil.mean);
  off = writeMatrix(view, off, foil.U);
  off = writeF32(view, off, foil.sigma);
  off = writeMatrix(view, off, foil.codes);

  if (off !== buf.byteLength) {
    throw new Error(`encodeFaceModelPack: length mismatch ${off} vs ${buf.byteLength}`);
  }
  return buf;
}

export function decodeFaceModelPack(
  buf: ArrayBuffer,
  examples: FaceExample[],
): { model: FaceModel; foil: PixelFoilModel } {
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== FACE_MODEL_PACK_MAGIC) throw new Error("face model pack: bad magic");
  const version = view.getUint32(4, true);
  if (version !== FACE_MODEL_PACK_VERSION) {
    throw new Error(`face model pack: unsupported version ${version}`);
  }
  const n = view.getUint32(8, true);
  const kApp = view.getUint32(12, true);
  const kShape = view.getUint32(16, true);
  const kFoil = view.getUint32(20, true);
  const size = view.getUint32(24, true);
  const nLandmarks = view.getUint32(28, true);
  const nTriIdx = view.getUint32(32, true);
  if (examples.length !== n) {
    throw new Error(`face model pack: example count ${examples.length} ≠ ${n}`);
  }

  const nPix = size * size;
  const shapeDim = nLandmarks * 2;
  let off = 36;

  let meanAppearance: Float64Array;
  ({ values: meanAppearance, next: off } = readF32(view, off, nPix));
  let appearanceU: Matrix;
  ({ matrix: appearanceU, next: off } = readMatrix(view, off, nPix, kApp));
  let appearanceSigmaArr: Float64Array;
  ({ values: appearanceSigmaArr, next: off } = readF32(view, off, kApp));
  let appearanceCodes: Matrix;
  ({ matrix: appearanceCodes, next: off } = readMatrix(view, off, n, kApp));

  let meanFlat: Float64Array;
  ({ values: meanFlat, next: off } = readF32(view, off, shapeDim));
  let shapeU: Matrix;
  ({ matrix: shapeU, next: off } = readMatrix(view, off, shapeDim, kShape));
  let shapeSigmaArr: Float64Array;
  ({ values: shapeSigmaArr, next: off } = readF32(view, off, kShape));
  let shapeCodes: Matrix;
  ({ matrix: shapeCodes, next: off } = readMatrix(view, off, n, kShape));

  const triangles = new Uint32Array(nTriIdx);
  for (let i = 0; i < nTriIdx; i++) triangles[i] = view.getUint32(off + 4 * i, true);
  off += 4 * nTriIdx;

  let foilMean: Float64Array;
  ({ values: foilMean, next: off } = readF32(view, off, nPix));
  let foilU: Matrix;
  ({ matrix: foilU, next: off } = readMatrix(view, off, nPix, kFoil));
  let foilSigmaArr: Float64Array;
  ({ values: foilSigmaArr, next: off } = readF32(view, off, kFoil));
  let foilCodes: Matrix;
  ({ matrix: foilCodes, next: off } = readMatrix(view, off, n, kFoil));

  const meanShape: Shape = unflattenShape(meanFlat);

  const model: FaceModel = {
    size,
    meanAppearance,
    appearanceU,
    appearanceSigma: Array.from(appearanceSigmaArr),
    appearanceCodes,
    meanShape,
    shapeU,
    shapeSigma: Array.from(shapeSigmaArr),
    shapeCodes,
    triangles,
    examples,
  };

  const foil: PixelFoilModel = {
    mean: foilMean,
    U: foilU,
    sigma: Array.from(foilSigmaArr),
    codes: foilCodes,
  };

  return { model, foil };
}

export async function fetchFaceModelPack(
  url: string,
  examples: FaceExample[],
  onProgress?: (msg: string) => void,
): Promise<{ model: FaceModel; foil: PixelFoilModel }> {
  onProgress?.("Loading SVD model…");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`face model pack ${res.status}`);
  const buf = await res.arrayBuffer();
  onProgress?.("Unpacking SVD model…");
  return decodeFaceModelPack(buf, examples);
}
