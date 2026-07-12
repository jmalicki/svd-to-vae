/** Parse IMM Face Database ASF landmark files (relative x,y in [0,1]). */

export type Point = { x: number; y: number };

export function parseAsf(text: string): Point[] {
  const lines = text.split(/\r?\n/);
  let n = -1;
  const points: Point[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (n < 0) {
      const maybe = Number(line);
      if (Number.isFinite(maybe) && maybe > 0) {
        n = maybe;
      }
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

/** Convert relative ASF coords to pixel coords for an image of size w×h. */
export function toPixels(rel: Point[], w: number, h: number): Point[] {
  return rel.map((p) => ({ x: p.x * w, y: p.y * h }));
}
