/**
 * Resize a canvas for HiDPI without letting bitmap size feed back into layout.
 *
 * Browsers use the width/height attributes as the default CSS size. Reading
 * clientWidth, writing canvas.width = clientWidth * dpr, then reading again
 * doubles the size every paint — which is what broke the mirror-angle slider.
 */

const CSS_W = "cssW";
const CSS_H = "cssH";

export type HiDpiCanvas = {
  cssW: number;
  cssH: number;
  dpr: number;
  ctx: CanvasRenderingContext2D | null;
};

function readInitialCssSize(canvas: HTMLCanvasElement): {
  cssW: number;
  cssH: number;
} {
  // Prefer a previously locked CSS size (survives attribute overwrites).
  const lockedW = Number(canvas.dataset[CSS_W]);
  const lockedH = Number(canvas.dataset[CSS_H]);
  if (lockedW > 0 && lockedH > 0) {
    return { cssW: lockedW, cssH: lockedH };
  }

  // Style already set by us or the page.
  const styleW = parseFloat(canvas.style.width);
  const styleH = parseFloat(canvas.style.height);
  if (styleW > 0 && styleH > 0) {
    return { cssW: styleW, cssH: styleH };
  }

  // Layout size before we mutate attributes (first paint only is safe).
  const layoutW = canvas.clientWidth;
  const layoutH = canvas.clientHeight;
  if (layoutW > 0 && layoutH > 0) {
    return { cssW: layoutW, cssH: layoutH };
  }

  // Fall back to the HTML width/height attributes as authored (not bitmap).
  const attrW = Number(canvas.getAttribute("width")) || 200;
  const attrH = Number(canvas.getAttribute("height")) || attrW;
  return { cssW: attrW, cssH: attrH };
}

/**
 * Lock CSS display size, set backing-store pixels for devicePixelRatio, and
 * return a 2D context already scaled so drawing uses CSS pixels.
 */
export function prepareHiDpiCanvas(
  canvas: HTMLCanvasElement,
  devicePixelRatio = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1,
): HiDpiCanvas {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const { cssW, cssH } = readInitialCssSize(canvas);

  canvas.dataset[CSS_W] = String(cssW);
  canvas.dataset[CSS_H] = String(cssH);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;

  const bw = Math.round(cssW * dpr);
  const bh = Math.round(cssH * dpr);
  if (canvas.width !== bw) canvas.width = bw;
  if (canvas.height !== bh) canvas.height = bh;

  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return { cssW, cssH, dpr, ctx };
}
