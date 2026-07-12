/** Binary pack of pre-warped IMM face examples (build-time or in-browser). */

import type { Shape } from "./faceWarp";

export type PackedFaceExample = {
  id: string;
  thumb: Float64Array;
  appearance: Float64Array;
  shape: Shape;
};

export const IMM_PACK_MAGIC = 0x464d4d49; // "IMMF" little-endian
export const IMM_PACK_VERSION = 1;

function u8FromGray(gray: Float64Array): Uint8Array {
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.min(255, Math.max(0, Math.round(gray[i]! * 255)));
  }
  return out;
}

function grayFromU8(bytes: Uint8Array): Float64Array {
  const out = new Float64Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = bytes[i]! / 255;
  return out;
}

/**
 * Pack warped examples into one ArrayBuffer.
 * Layout (little-endian):
 *   magic u32, version u32, n u32, size u32, nLandmarks u32, idsBytes u32,
 *   ids UTF-8 JSON string[],
 *   appearances u8[n * size²],
 *   thumbs u8[n * size²],
 *   shapes f32[n * nLandmarks * 2]
 */
export function encodeImmPack(examples: PackedFaceExample[], size: number): ArrayBuffer {
  if (examples.length === 0) throw new Error("encodeImmPack: empty examples");
  const nLandmarks = examples[0]!.shape.length;
  const idsJson = new TextEncoder().encode(JSON.stringify(examples.map((e) => e.id)));
  const pix = size * size;
  const header = 24;
  const grayBytes = examples.length * pix * 2;
  const afterGray = header + idsJson.byteLength + grayBytes;
  const shapesOff = (afterGray + 3) & ~3; // Float32Array needs 4-byte alignment
  const shapesBytes = examples.length * nLandmarks * 2 * 4;
  const buf = new ArrayBuffer(shapesOff + shapesBytes);
  const view = new DataView(buf);
  view.setUint32(0, IMM_PACK_MAGIC, true);
  view.setUint32(4, IMM_PACK_VERSION, true);
  view.setUint32(8, examples.length, true);
  view.setUint32(12, size, true);
  view.setUint32(16, nLandmarks, true);
  view.setUint32(20, idsJson.byteLength, true);
  new Uint8Array(buf, header, idsJson.byteLength).set(idsJson);

  let off = header + idsJson.byteLength;
  const u8 = new Uint8Array(buf);
  for (const ex of examples) {
    if (ex.appearance.length !== pix || ex.thumb.length !== pix) {
      throw new Error(`encodeImmPack: bad gray length for ${ex.id}`);
    }
    if (ex.shape.length !== nLandmarks) {
      throw new Error(`encodeImmPack: landmark count mismatch for ${ex.id}`);
    }
    u8.set(u8FromGray(ex.appearance), off);
    off += pix;
  }
  for (const ex of examples) {
    u8.set(u8FromGray(ex.thumb), off);
    off += pix;
  }
  const f32 = new Float32Array(buf, shapesOff, examples.length * nLandmarks * 2);
  let fi = 0;
  for (const ex of examples) {
    for (const p of ex.shape) {
      f32[fi++] = p.x;
      f32[fi++] = p.y;
    }
  }
  return buf;
}

export function decodeImmPack(buf: ArrayBuffer): { size: number; examples: PackedFaceExample[] } {
  const view = new DataView(buf);
  const magic = view.getUint32(0, true);
  if (magic !== IMM_PACK_MAGIC) throw new Error("imm pack: bad magic");
  const version = view.getUint32(4, true);
  if (version !== IMM_PACK_VERSION) throw new Error(`imm pack: unsupported version ${version}`);
  const n = view.getUint32(8, true);
  const size = view.getUint32(12, true);
  const nLandmarks = view.getUint32(16, true);
  const idsBytes = view.getUint32(20, true);
  const header = 24;
  const ids = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, header, idsBytes))) as string[];
  if (ids.length !== n) throw new Error("imm pack: id count mismatch");

  const pix = size * size;
  let off = header + idsBytes;
  const appearances: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    appearances.push(grayFromU8(new Uint8Array(buf, off, pix)));
    off += pix;
  }
  const thumbs: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    thumbs.push(grayFromU8(new Uint8Array(buf, off, pix)));
    off += pix;
  }
  const shapesOff = (off + 3) & ~3;
  const f32 = new Float32Array(buf, shapesOff, n * nLandmarks * 2);
  const examples: PackedFaceExample[] = [];
  for (let i = 0; i < n; i++) {
    const shape: Shape = [];
    const base = i * nLandmarks * 2;
    for (let j = 0; j < nLandmarks; j++) {
      shape.push({ x: f32[base + 2 * j]!, y: f32[base + 2 * j + 1]! });
    }
    examples.push({
      id: ids[i]!,
      appearance: appearances[i]!,
      thumb: thumbs[i]!,
      shape,
    });
  }
  return { size, examples };
}

export async function fetchImmPack(
  url: string,
  onProgress?: (msg: string) => void,
): Promise<{ size: number; examples: PackedFaceExample[] }> {
  onProgress?.("Loading face pack…");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`imm pack ${res.status}`);
  const buf = await res.arrayBuffer();
  onProgress?.("Unpacking faces…");
  return decodeImmPack(buf);
}
