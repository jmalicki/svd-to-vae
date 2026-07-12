/**
 * Child task: build face model or pixel-foil SVD and v8-serialize to disk.
 * Usage: vite-node scripts/bake-svd-task.ts <face|foil> <examples.v8> <out.v8> <rank>
 */
import { readFileSync, writeFileSync } from "node:fs";
import v8 from "node:v8";
import { buildFaceModel, buildPixelFoilModel, type FaceExample } from "../src/faceModel";

const task = process.argv[2];
const examplesPath = process.argv[3];
const outPath = process.argv[4];
const rank = Number(process.argv[5]);
if (task !== "face" && task !== "foil") {
  console.error("usage: bake-svd-task.ts <face|foil> <examples.v8> <out.v8> <rank>");
  process.exit(2);
}

const examples = v8.deserialize(readFileSync(examplesPath)) as FaceExample[];
const t0 = performance.now();
const result =
  task === "face" ? buildFaceModel(examples, rank) : buildPixelFoilModel(examples, rank);
if (task === "face") {
  (result as { examples: unknown }).examples = [];
}
writeFileSync(outPath, v8.serialize(result));
console.log(`  ${task} SVD ${(performance.now() - t0).toFixed(0)}ms → ${outPath}`);
