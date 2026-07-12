import { describe, expect, it } from "vitest";
import { wrapAngleDeg } from "./angleDial";

describe("wrapAngleDeg", () => {
  it("wraps a full turn into (−180, 180]", () => {
    expect(wrapAngleDeg(0, 360)).toBeCloseTo(0, 10);
    expect(wrapAngleDeg(180, 360)).toBeCloseTo(180, 10);
    expect(wrapAngleDeg(-180, 360)).toBeCloseTo(180, 10);
    expect(wrapAngleDeg(190, 360)).toBeCloseTo(-170, 10);
    expect(wrapAngleDeg(370, 360)).toBeCloseTo(10, 10);
    expect(wrapAngleDeg(-190, 360)).toBeCloseTo(170, 10);
  });

  it("wraps a half-turn (mirror line) into (−90, 90]", () => {
    expect(wrapAngleDeg(0, 180)).toBeCloseTo(0, 10);
    expect(wrapAngleDeg(90, 180)).toBeCloseTo(90, 10);
    expect(wrapAngleDeg(-90, 180)).toBeCloseTo(90, 10);
    expect(wrapAngleDeg(100, 180)).toBeCloseTo(-80, 10);
    expect(wrapAngleDeg(270, 180)).toBeCloseTo(90, 10);
  });

  it("is continuous when dragging past the branch cut", () => {
    // Crossing +90° for a line should land near −90°, not jump by 180° of geometry.
    const a = wrapAngleDeg(89, 180);
    const b = wrapAngleDeg(91, 180);
    expect(a).toBeCloseTo(89, 10);
    expect(b).toBeCloseTo(-89, 10);
    // Same undirected line as 91° − 180° = −89°.
  });
});
