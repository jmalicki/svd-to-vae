import { describe, expect, it } from "vitest";
import { encodeImmPack, decodeImmPack } from "./immPack";

describe("immPack", () => {
  it("round-trips warped examples", () => {
    const size = 4;
    const examples = [
      {
        id: "01-1m",
        appearance: Float64Array.from([0, 0.25, 0.5, 0.75, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.35]),
        thumb: Float64Array.from(Array.from({ length: 16 }, (_, i) => i / 15)),
        shape: [
          { x: 1.5, y: 2.25 },
          { x: 3, y: 0.5 },
        ],
      },
      {
        id: "02-1f",
        appearance: Float64Array.from(Array.from({ length: 16 }, () => 0.42)),
        thumb: Float64Array.from(Array.from({ length: 16 }, () => 0.7)),
        shape: [
          { x: 0, y: 0 },
          { x: 3.5, y: 3.5 },
        ],
      },
    ];
    const buf = encodeImmPack(examples, size);
    const { size: outSize, examples: out } = decodeImmPack(buf);
    expect(outSize).toBe(size);
    expect(out).toHaveLength(2);
    expect(out[0]!.id).toBe("01-1m");
    expect(out[1]!.shape[1]).toEqual({ x: 3.5, y: 3.5 });
    for (let i = 0; i < 16; i++) {
      expect(out[0]!.appearance[i]).toBeCloseTo(examples[0]!.appearance[i]!, 2);
      expect(out[0]!.thumb[i]).toBeCloseTo(examples[0]!.thumb[i]!, 2);
    }
  });
});
