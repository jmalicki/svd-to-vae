/**
 * Bake pre-warped IMM faces + SVD models for fast page load.
 *
 * Parallelism:
 *   - JPEG/ASF decode across worker_threads
 *   - face SVD ∥ foil SVD in child vite-node processes (CPU-bound)
 *
 * Run: npm run gen:imm-pack
 */
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import v8 from "node:v8";
import {
  FACE_SIZE,
  buildExamplesFromLoaded,
  type FaceExample,
  type FaceModel,
  type ImmManifest,
  type LoadedImmFace,
} from "../src/faceModel";
import { encodeImmPack } from "../src/immPack";
import { encodeFaceModelPack, type PixelFoilModel } from "../src/faceModelPack";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(__dirname, "..");
const IMM_DIR = path.resolve(WEB_ROOT, "public/imm");
const OUT_EXAMPLES = path.resolve(IMM_DIR, "examples.bin");
const OUT_MODEL = path.resolve(IMM_DIR, "model.bin");
const DECODE_WORKER = path.resolve(__dirname, "imm-decode-worker.mjs");
const VITE_NODE = path.resolve(WEB_ROOT, "node_modules/vite-node/vite-node.mjs");

function chunkIds<T>(arr: T[], nChunks: number): T[][] {
  const n = Math.max(1, Math.min(nChunks, arr.length));
  const chunks: T[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < arr.length; i++) chunks[i % n]!.push(arr[i]!);
  return chunks.filter((c) => c.length > 0);
}

function decodeBatch(ids: string[]): Promise<LoadedImmFace[]> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(DECODE_WORKER, {
      workerData: { immDir: IMM_DIR, ids },
    });
    worker.on("message", (rows: Array<{
      id: string;
      width: number;
      height: number;
      data: Uint8ClampedArray;
      shapePx: { x: number; y: number }[];
    }>) => {
      resolve(
        rows.map((r) => ({
          id: r.id,
          img: { width: r.width, height: r.height, data: r.data },
          shapePx: r.shapePx,
        })),
      );
    });
    worker.on("error", reject);
    worker.on("exit", (code) => {
      if (code !== 0) reject(new Error(`decode worker exited ${code}`));
    });
  });
}

async function decodeAllParallel(ids: string[]): Promise<LoadedImmFace[]> {
  const workers = Math.min(os.cpus().length, 8, ids.length);
  const chunks = chunkIds(ids, workers);
  console.log(`  decoding with ${chunks.length} workers…`);
  const t0 = performance.now();
  const batches = await Promise.all(chunks.map((c) => decodeBatch(c)));
  // Preserve manifest order
  const byId = new Map<string, LoadedImmFace>();
  for (const batch of batches) for (const face of batch) byId.set(face.id, face);
  const ordered = ids.map((id) => {
    const f = byId.get(id);
    if (!f) throw new Error(`missing decode for ${id}`);
    return f;
  });
  console.log(`  decoded ${ordered.length} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return ordered;
}

function runViteNode(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [VITE_NODE, ...args], {
      cwd: WEB_ROOT,
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`vite-node ${args[0]} exited ${code}`));
    });
  });
}

async function buildSvdsParallel(
  examples: FaceExample[],
  rank: number,
): Promise<{ model: FaceModel; foil: PixelFoilModel }> {
  const examplesPath = path.join(os.tmpdir(), `imm-examples-${process.pid}.v8`);
  const faceOut = path.join(os.tmpdir(), `imm-face-${process.pid}.v8`);
  const foilOut = path.join(os.tmpdir(), `imm-foil-${process.pid}.v8`);
  const task = path.resolve(__dirname, "bake-svd-task.ts");
  writeFileSync(examplesPath, v8.serialize(examples));
  const t0 = performance.now();
  console.log(`  SVD rank ${rank} (face ∥ foil)…`);
  try {
    await Promise.all([
      runViteNode([task, "face", examplesPath, faceOut, String(rank)]),
      runViteNode([task, "foil", examplesPath, foilOut, String(rank)]),
    ]);
    const model = v8.deserialize(readFileSync(faceOut)) as FaceModel;
    const foil = v8.deserialize(readFileSync(foilOut)) as PixelFoilModel;
    console.log(`  SVD wall ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    return { model, foil };
  } finally {
    for (const p of [examplesPath, faceOut, foilOut]) {
      try {
        unlinkSync(p);
      } catch {
        /* ignore */
      }
    }
  }
}

const manifest = JSON.parse(readFileSync(path.join(IMM_DIR, "manifest.json"), "utf8")) as ImmManifest;
const t0 = performance.now();
console.log(`Baking ${manifest.files.length} IMM faces`);

const loaded = await decodeAllParallel(manifest.files);

const examples = await buildExamplesFromLoaded(loaded, FACE_SIZE, (msg) => {
  if (msg.startsWith("Warping")) {
    const m = /^Warping (\d+)\/(\d+)/.exec(msg);
    if (m && Number(m[1]) % 40 !== 0 && Number(m[1]) !== Number(m[2])) return;
  }
  console.log(`  ${msg}`);
});

const exBuf = encodeImmPack(examples, FACE_SIZE);
writeFileSync(OUT_EXAMPLES, Buffer.from(exBuf));
console.log(
  `  wrote ${path.relative(WEB_ROOT, OUT_EXAMPLES)} (${(exBuf.byteLength / (1024 * 1024)).toFixed(2)} MiB)`,
);

const fullK = Math.min(examples.length - 1, FACE_SIZE * FACE_SIZE);
const { model, foil } = await buildSvdsParallel(examples, fullK);
model.examples = examples;

const modelBuf = encodeFaceModelPack(model, foil);
writeFileSync(OUT_MODEL, Buffer.from(modelBuf));
console.log(
  `  wrote ${path.relative(WEB_ROOT, OUT_MODEL)} (${(modelBuf.byteLength / (1024 * 1024)).toFixed(2)} MiB)`,
);
console.log(`Bake wall time ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const { spawnSync } = await import("node:child_process");
const check = spawnSync("npm", ["run", "test:imm-pack"], {
  stdio: "inherit",
  cwd: WEB_ROOT,
});
if (check.status !== 0) process.exit(check.status ?? 1);
