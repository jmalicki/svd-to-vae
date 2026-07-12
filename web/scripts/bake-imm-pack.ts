/**
 * Bake pre-warped IMM faces + SVD models for fast page load.
 *
 * Uses the same TS pipeline as the browser (`buildExamplesFromLoaded`,
 * `buildFaceModel`, `encodeImmPack`, `encodeFaceModelPack`). JPEG decode is
 * pure JS (`jpeg-js`) so this could run in-page too — build time is just
 * “do it once.”
 *
 * Run: npm run gen:imm-pack
 */
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import {
  FACE_SIZE,
  buildExamplesFromLoaded,
  buildFaceModel,
  buildPixelFoilModel,
  rgbaToGrayInPlace,
  type ImmManifest,
  type LoadedImmFace,
} from "../src/faceModel";
import { encodeImmPack } from "../src/immPack";
import { encodeFaceModelPack } from "../src/faceModelPack";
import { parseAsf, toPixels } from "../src/immAsf";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMM_DIR = path.resolve(__dirname, "../public/imm");
const OUT_EXAMPLES = path.resolve(IMM_DIR, "examples.bin");
const OUT_MODEL = path.resolve(IMM_DIR, "model.bin");

function loadOneFromDisk(id: string): LoadedImmFace {
  const jpg = readFileSync(path.join(IMM_DIR, `${id}.jpg`));
  const asfText = readFileSync(path.join(IMM_DIR, `${id}.asf`), "utf8");
  const decoded = jpeg.decode(jpg, { useTArray: true });
  const img = {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
  rgbaToGrayInPlace(img);
  const shapePx = toPixels(parseAsf(asfText), img.width, img.height);
  return { id, img, shapePx };
}

const manifest = JSON.parse(readFileSync(path.join(IMM_DIR, "manifest.json"), "utf8")) as ImmManifest;
const t0 = performance.now();
console.log(`Baking ${manifest.files.length} IMM faces`);

const loaded: LoadedImmFace[] = [];
for (let i = 0; i < manifest.files.length; i++) {
  const id = manifest.files[i]!;
  loaded.push(loadOneFromDisk(id));
  if ((i + 1) % 40 === 0 || i + 1 === manifest.files.length) {
    console.log(`  decoded ${i + 1}/${manifest.files.length}`);
  }
}

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
  `  wrote ${path.relative(process.cwd(), OUT_EXAMPLES)} (${(exBuf.byteLength / (1024 * 1024)).toFixed(2)} MiB)`,
);

const fullK = Math.min(examples.length - 1, FACE_SIZE * FACE_SIZE);
console.log(`  SVD rank ${fullK}…`);
const tSvd = performance.now();
const model = buildFaceModel(examples, fullK);
const foil = buildPixelFoilModel(examples, fullK);
console.log(`  SVD done in ${((performance.now() - tSvd) / 1000).toFixed(1)}s`);

const modelBuf = encodeFaceModelPack(model, foil);
writeFileSync(OUT_MODEL, Buffer.from(modelBuf));
console.log(
  `  wrote ${path.relative(process.cwd(), OUT_MODEL)} (${(modelBuf.byteLength / (1024 * 1024)).toFixed(2)} MiB)`,
);
console.log(`Bake wall time ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const { spawnSync } = await import("node:child_process");
const check = spawnSync("npm", ["run", "test:imm-pack"], {
  stdio: "inherit",
  cwd: path.resolve(__dirname, ".."),
  shell: true,
});
if (check.status !== 0) process.exit(check.status ?? 1);