import { describe, expect, it } from "vitest";
import { prepareHiDpiCanvas } from "./hiDpiCanvas";

/**
 * Minimal canvas stand-in that mimics the browser feedback loop:
 * with no CSS width, clientWidth tracks the width/height properties.
 */
function makeFeedbackCanvas(
  initialW = 300,
  initialH = initialW,
): HTMLCanvasElement {
  const style: { width: string; height: string } = { width: "", height: "" };
  const dataset: Record<string, string> = {};
  let width = initialW;
  let height = initialH;
  const attrs: Record<string, string> = {
    width: String(initialW),
    height: String(initialH),
  };

  const canvas = {
    style,
    dataset,
    get width() {
      return width;
    },
    set width(v: number) {
      width = v;
      attrs.width = String(v);
    },
    get height() {
      return height;
    },
    set height(v: number) {
      height = v;
      attrs.height = String(v);
    },
    get clientWidth() {
      if (style.width) return parseFloat(style.width) || 0;
      return width;
    },
    get clientHeight() {
      if (style.height) return parseFloat(style.height) || 0;
      return height;
    },
    getAttribute(name: string) {
      return attrs[name] ?? null;
    },
    getContext() {
      return {
        setTransform() {},
        clearRect() {},
        fillRect() {},
      };
    },
  };
  return canvas as unknown as HTMLCanvasElement;
}

describe("prepareHiDpiCanvas", () => {
  it("does not grow across repeated paints (mirror-slider re-draw bug)", () => {
    const canvas = makeFeedbackCanvas(300);
    const sizes: number[] = [];
    for (let i = 0; i < 12; i++) {
      const { cssW, dpr } = prepareHiDpiCanvas(canvas, 2);
      sizes.push(cssW);
      expect(canvas.width).toBe(Math.round(cssW * dpr));
      expect(canvas.clientWidth).toBe(300);
    }
    expect([...new Set(sizes)]).toEqual([300]);
    expect(canvas.width).toBe(600);
  });

  it("keeps non-square stem canvases stable", () => {
    const canvas = makeFeedbackCanvas(220, 160);
    for (let i = 0; i < 8; i++) {
      const { cssW, cssH, dpr } = prepareHiDpiCanvas(canvas, 2);
      expect(cssW).toBe(220);
      expect(cssH).toBe(160);
      expect(canvas.width).toBe(Math.round(220 * dpr));
      expect(canvas.height).toBe(Math.round(160 * dpr));
      expect(canvas.clientWidth).toBe(220);
      expect(canvas.clientHeight).toBe(160);
    }
  });

  it("recovers from a previously blown-up bitmap when CSS size is locked", () => {
    const canvas = makeFeedbackCanvas(300);
    canvas.dataset.cssW = "300";
    canvas.dataset.cssH = "300";
    canvas.style.width = "300px";
    canvas.style.height = "300px";
    canvas.width = 76800;
    canvas.height = 76800;

    const { cssW, cssH, dpr } = prepareHiDpiCanvas(canvas, 2);
    expect(cssW).toBe(300);
    expect(cssH).toBe(300);
    expect(canvas.width).toBe(Math.round(300 * dpr));
    expect(canvas.clientWidth).toBe(300);
  });

  it("without the CSS lock, naive clientWidth*dpr sizing would explode", () => {
    // Document the failure mode so a regression is obvious.
    const canvas = makeFeedbackCanvas(300);
    let css = canvas.clientWidth;
    for (let i = 0; i < 8; i++) {
      const dpr = 2;
      canvas.width = Math.round(css * dpr);
      canvas.height = Math.round(css * dpr);
      css = canvas.clientWidth; // grows: 300 → 600 → 1200 → …
    }
    expect(canvas.clientWidth).toBeGreaterThan(10_000);
  });
});
