/**
 * Worker: decode a batch of IMM jpg+asf pairs from disk.
 * Plain JS so node:worker_threads can load it without a TS loader.
 */
import { parentPort, workerData } from "node:worker_threads";
import { readFileSync } from "node:fs";
import path from "node:path";
import jpeg from "jpeg-js";

function parseAsf(text) {
  const lines = text.split(/\r?\n/);
  let n = -1;
  const points = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (n < 0) {
      const maybe = Number(line);
      if (Number.isFinite(maybe) && maybe > 0) n = maybe;
      continue;
    }
    if (points.length >= n) break;
    const parts = line.split(/\s+/);
    if (parts.length < 4) continue;
    const x = Number(parts[2]);
    const y = Number(parts[3]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    points.push({ x, y });
  }
  if (n > 0 && points.length !== n) {
    throw new Error(`ASF expected ${n} points, got ${points.length}`);
  }
  return points;
}

function rgbaToGrayInPlace(data) {
  for (let i = 0; i < data.length; i += 4) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = data[i + 1] = data[i + 2] = y;
  }
}

const { immDir, ids } = workerData;
const out = [];
for (const id of ids) {
  const jpg = readFileSync(path.join(immDir, `${id}.jpg`));
  const asfText = readFileSync(path.join(immDir, `${id}.asf`), "utf8");
  const decoded = jpeg.decode(jpg, { useTArray: true });
  const data = new Uint8ClampedArray(decoded.data);
  rgbaToGrayInPlace(data);
  const rel = parseAsf(asfText);
  const shapePx = rel.map((p) => ({ x: p.x * decoded.width, y: p.y * decoded.height }));
  out.push({
    id,
    width: decoded.width,
    height: decoded.height,
    data,
    shapePx,
  });
}
parentPort.postMessage(out);
