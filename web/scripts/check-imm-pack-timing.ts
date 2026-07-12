/**
 * Assert prebaked IMM packs unpack quickly (the page-load path).
 * Fails if examples.bin / model.bin are missing or decode is slow —
 * that usually means someone reintroduced runtime SVD.
 *
 * Run: npm run test:imm-pack
 * Also invoked at the end of `npm run gen:imm-pack`.
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodeImmPack } from "../src/immPack";
import { decodeFaceModelPack } from "../src/faceModelPack";
import { decodeAppearance, getAppearanceCode, mseGray } from "../src/faceModel";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMM_DIR = path.resolve(__dirname, "../public/imm");
const EXAMPLES = path.join(IMM_DIR, "examples.bin");
const MODEL = path.join(IMM_DIR, "model.bin");

/** CI runners are slower than a laptop; still far below a full SVD (~10s+). */
const MAX_UNPACK_MS = 3000;
/** Rank-k rebuild of a training face should nearly match at full model rank. */
const MAX_RECON_MSE = 1e-3;

function readBuf(p: string): ArrayBuffer {
  const raw = readFileSync(p);
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
}

if (!existsSync(EXAMPLES) || !existsSync(MODEL)) {
  console.error("Missing public/imm/examples.bin or model.bin — run npm run gen:imm-pack first.");
  process.exit(1);
}

const exBytes = readFileSync(EXAMPLES).byteLength;
const modelBytes = readFileSync(MODEL).byteLength;
console.log(
  `Pack sizes: examples ${(exBytes / (1024 * 1024)).toFixed(2)} MiB, model ${(modelBytes / (1024 * 1024)).toFixed(2)} MiB`,
);

const t0 = performance.now();
const { examples } = decodeImmPack(readBuf(EXAMPLES));
const tEx = performance.now();
const { model } = decodeFaceModelPack(readBuf(MODEL), examples);
const tDone = performance.now();

const unpackMs = tDone - t0;
console.log(
  `Unpack: examples ${(tEx - t0).toFixed(0)}ms + model ${(tDone - tEx).toFixed(0)}ms = ${unpackMs.toFixed(0)}ms total (${examples.length} faces, k=${model.appearanceU.cols})`,
);

if (unpackMs > MAX_UNPACK_MS) {
  console.error(
    `FAIL: unpack took ${unpackMs.toFixed(0)}ms > ${MAX_UNPACK_MS}ms budget. Page load must stay on the pack path (no runtime SVD).`,
  );
  process.exit(1);
}

const code = getAppearanceCode(model, 0);
const recon = decodeAppearance(model, code, model.appearanceU.cols);
const err = mseGray(model.examples[0]!.appearance, recon);
console.log(`Full-rank recon MSE (face 0): ${err.toExponential(2)}`);
if (err > MAX_RECON_MSE) {
  console.error(`FAIL: full-rank reconstruction MSE ${err} > ${MAX_RECON_MSE}`);
  process.exit(1);
}

console.log(`OK: pack load within ${MAX_UNPACK_MS}ms budget`);
