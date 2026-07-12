import { describe, expect, it } from "vitest";
import { mat, set } from "./matrix";
import { encodeFaceModelPack, decodeFaceModelPack } from "./faceModelPack";
import type { FaceExample, FaceModel } from "./faceModel";
import type { PixelFoilModel } from "./faceModelPack";

describe("faceModelPack", () => {
  it("round-trips a tiny model + foil", () => {
    const size = 2;
    const nPix = size * size;
    const examples: FaceExample[] = [
      {
        id: "a",
        appearance: Float64Array.from([0.1, 0.2, 0.3, 0.4]),
        thumb: Float64Array.from([0.5, 0.6, 0.7, 0.8]),
        shape: [
          { x: 1, y: 2 },
          { x: 3, y: 4 },
        ],
      },
      {
        id: "b",
        appearance: Float64Array.from([0.2, 0.3, 0.4, 0.5]),
        thumb: Float64Array.from([0.1, 0.2, 0.3, 0.4]),
        shape: [
          { x: 1.5, y: 2.5 },
          { x: 3.5, y: 4.5 },
        ],
      },
    ];
    const appearanceU = mat(nPix, 1);
    set(appearanceU, 0, 0, 1);
    const appearanceCodes = mat(2, 1);
    set(appearanceCodes, 0, 0, 0.5);
    set(appearanceCodes, 1, 0, -0.25);
    const shapeU = mat(4, 1);
    set(shapeU, 0, 0, 1);
    const shapeCodes = mat(2, 1);
    const model: FaceModel = {
      size,
      meanAppearance: Float64Array.from([0.25, 0.25, 0.25, 0.25]),
      appearanceU,
      appearanceSigma: [2],
      appearanceCodes,
      meanShape: [
        { x: 1.25, y: 2.25 },
        { x: 3.25, y: 4.25 },
      ],
      shapeU,
      shapeSigma: [1],
      shapeCodes,
      triangles: new Uint32Array([0, 1, 0]),
      examples,
    };
    const foilU = mat(nPix, 1);
    set(foilU, 1, 0, 1);
    const foil: PixelFoilModel = {
      mean: Float64Array.from([0.4, 0.4, 0.4, 0.4]),
      U: foilU,
      sigma: [3],
      codes: appearanceCodes,
    };
    const buf = encodeFaceModelPack(model, foil);
    const out = decodeFaceModelPack(buf, examples);
    expect(out.model.appearanceSigma[0]).toBeCloseTo(2, 5);
    expect(out.model.meanShape[0]).toEqual({ x: 1.25, y: 2.25 });
    expect(out.foil.sigma[0]).toBeCloseTo(3, 5);
    expect(out.model.triangles[1]).toBe(1);
  });
});
