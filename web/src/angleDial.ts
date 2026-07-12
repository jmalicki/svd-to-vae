/** Circular angle dial — wraps naturally, unlike a linear slider. */

export type AngleDialKind = "line" | "ray";

export type AngleDialOptions = {
  /** Angular period in degrees (360 for a directed ray, 180 for an undirected line). */
  periodDeg?: number;
  /** Initial angle in degrees (math convention: 0 = +x, positive = CCW). */
  valueDeg: number;
  color: string;
  kind: AngleDialKind;
  onChange: (deg: number) => void;
};

/** Fold angle into (−period/2, period/2]. */
export function wrapAngleDeg(deg: number, periodDeg: number): number {
  const p = periodDeg;
  let x = ((((deg + p / 2) % p) + p) % p) - p / 2;
  if (x <= -p / 2 + 1e-12) x = p / 2;
  return x;
}

function pointerAngleDeg(
  canvas: HTMLCanvasElement,
  clientX: number,
  clientY: number,
): number {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left - rect.width / 2;
  const y = clientY - rect.top - rect.height / 2;
  // Screen y grows down; math angles grow CCW with y up.
  return (Math.atan2(-y, x) * 180) / Math.PI;
}

export type AngleDial = {
  get: () => number;
  set: (deg: number, silent?: boolean) => void;
  redraw: () => void;
};

/**
 * Bind a small canvas as a draggable angle dial.
 * `kind: "line"` draws a diameter (mirror); `kind: "ray"` draws an arrow (probe).
 */
export function mountAngleDial(
  canvas: HTMLCanvasElement,
  opts: AngleDialOptions,
): AngleDial {
  const period = opts.periodDeg ?? (opts.kind === "line" ? 180 : 360);
  let value = wrapAngleDeg(opts.valueDeg, period);
  let dragging = false;

  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

  function sizeCss(): number {
    return canvas.clientWidth || Number(canvas.getAttribute("width")) || 120;
  }

  function redraw(): void {
    const css = sizeCss();
    const ratio = dpr();
    // Lock CSS size so HiDPI bitmap updates don’t grow the layout.
    if (!canvas.style.width) {
      canvas.style.width = `${css}px`;
      canvas.style.height = `${css}px`;
    }
    const cssW = parseFloat(canvas.style.width) || css;
    canvas.width = Math.round(cssW * ratio);
    canvas.height = Math.round(cssW * ratio);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    const w = cssW;
    const cx = w / 2;
    const cy = w / 2;
    const r = w * 0.38;

    ctx.clearRect(0, 0, w, w);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, w);

    // Ring
    ctx.strokeStyle = "rgba(45,45,45,0.18)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Light axes
    ctx.strokeStyle = "rgba(45,45,45,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.stroke();

    const th = (value * Math.PI) / 180;
    const ux = Math.cos(th);
    const uy = Math.sin(th);

    ctx.strokeStyle = opts.color;
    ctx.fillStyle = opts.color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";

    if (opts.kind === "line") {
      const nx = -uy;
      const ny = ux;
      const band = 4;
      ctx.fillStyle = "rgba(213, 94, 0, 0.14)";
      ctx.beginPath();
      ctx.moveTo(cx - r * ux + nx * band, cy + r * uy - ny * band);
      ctx.lineTo(cx + r * ux + nx * band, cy - r * uy - ny * band);
      ctx.lineTo(cx + r * ux - nx * band, cy - r * uy + ny * band);
      ctx.lineTo(cx - r * ux - nx * band, cy + r * uy + ny * band);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = opts.color;
      ctx.beginPath();
      ctx.moveTo(cx - r * ux, cy + r * uy);
      ctx.lineTo(cx + r * ux, cy - r * uy);
      ctx.stroke();
      ctx.fillStyle = opts.color;
      ctx.beginPath();
      ctx.arc(cx + r * ux, cy - r * uy, 5.5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const tipX = cx + r * ux;
      const tipY = cy - r * uy;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      const head = 8;
      const hx = ux;
      const hy = -uy; // screen direction of the ray
      const len = Math.hypot(hx, hy) || 1;
      const sx = hx / len;
      const sy = hy / len;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - head * sx + 0.5 * head * sy, tipY - head * sy - 0.5 * head * sx);
      ctx.lineTo(tipX - head * sx - 0.5 * head * sy, tipY - head * sy + 0.5 * head * sx);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(tipX, tipY, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Center
    ctx.fillStyle = "rgba(45,45,45,0.45)";
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  function setFromPointer(clientX: number, clientY: number): void {
    let deg = pointerAngleDeg(canvas, clientX, clientY);
    if (opts.kind === "line") {
      // Undirected: fold into half-turn so opposite rays are the same line.
      deg = wrapAngleDeg(deg, period);
    } else {
      deg = wrapAngleDeg(deg, period);
    }
    if (Math.abs(deg - value) < 1e-9) return;
    value = deg;
    redraw();
    opts.onChange(value);
  }

  canvas.style.touchAction = "none";
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "slider");
  canvas.setAttribute("aria-valuemin", String(-period / 2));
  canvas.setAttribute("aria-valuemax", String(period / 2));
  canvas.setAttribute("aria-valuenow", String(Math.round(value)));

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    setFromPointer(e.clientX, e.clientY);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    setFromPointer(e.clientX, e.clientY);
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener("pointercancel", () => {
    dragging = false;
  });
  canvas.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 5 : 1;
    let next = value;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next -= step;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") next += step;
    else return;
    e.preventDefault();
    value = wrapAngleDeg(next, period);
    canvas.setAttribute("aria-valuenow", String(Math.round(value)));
    redraw();
    opts.onChange(value);
  });

  redraw();

  return {
    get: () => value,
    set: (deg, silent = false) => {
      value = wrapAngleDeg(deg, period);
      canvas.setAttribute("aria-valuenow", String(Math.round(value)));
      redraw();
      if (!silent) opts.onChange(value);
    },
    redraw,
  };
}
