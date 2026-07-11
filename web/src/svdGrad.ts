import { torch } from "js-pytorch";
import {
  type Matrix,
  mat,
  toNested,
  fromTensorData,
  frobeniusSq,
  reconstruct,
  sub,
  thinQ,
  writeIntoNested,
  maxAbs,
  copy,
} from "./matrix";

export type DeviceKind = "cpu" | "gpu";

export type LossParts = {
  recon: number;
};

export type GradState = {
  U: Matrix;
  sigma: number[];
  V: Matrix;
  loss: LossParts;
  step: number;
  rank: number;
  device: DeviceKind;
  /** Current adaptive step size (scalar; same for all parameters). */
  lr: number;
};

/** Adam-style second-moment EMA coefficient for the global scalar v. */
export const LR_BETA2 = 0.999;
/** Armijo sufficient-decrease constant. */
export const ARMIJO_C = 1e-4;
/** Max backtracking shrinks of α (α ← α/2). */
export const ARMIJO_MAX_BT = 12;

type AnyTensor = {
  data: unknown;
  shape: number[];
  _data?: unknown;
  _grad?: AnyTensor | null;
  add: (o: AnyTensor | number) => AnyTensor;
  sub: (o: AnyTensor | number) => AnyTensor;
  mul: (o: AnyTensor | number) => AnyTensor;
  pow: (n: number) => AnyTensor;
  exp: () => AnyTensor;
  log: () => AnyTensor;
  sum: (dim?: number) => AnyTensor;
  matmul: (o: AnyTensor) => AnyTensor;
  transpose: (a: number, b: number) => AnyTensor;
  backward: () => void;
  zero_grad?: () => void;
};

/**
 * Float64 noise floor for mean(g²), matched to the demo’s reconstruction ULP idea:
 * per-entry residual noise ~ ε·‖A‖_∞, so smaller gradient energy is not meaningful.
 */
export function gradSecondMomentFpFloor(dataScale: number): number {
  const ulp = Number.EPSILON * Math.max(dataScale, Number.EPSILON);
  return ulp * ulp;
}

/**
 * SGD with a single Adam-style second moment on mean(g²).
 * One scalar v → one η for every parameter. No per-entry buffers to fight QR.
 *
 *   ḡ² ← mean_i g_i²
 *   v  ← β₂ v + (1−β₂) ḡ²
 *   v† ← max(v̂, ḡ², v_fp)
 *   η  ← η₀ / (√v† + √v_fp)
 *
 * The trainer then backtracks α∈{1,½,…} on the QR retract so the MSE decreases
 * (Armijo), which keeps a reached Eckart–Young Â from walking away.
 */
class GlobalRmsSgd {
  params: AnyTensor[];
  baseLr: number;
  beta2: number;
  vFp: number;
  v: number;
  t: number;
  lr: number;
  /** ‖g‖₂² from the last adaptLr call (for Armijo). */
  lastSumSq: number;

  constructor(params: AnyTensor[], baseLr: number, dataScale: number) {
    this.params = params;
    this.baseLr = baseLr;
    this.beta2 = LR_BETA2;
    this.vFp = gradSecondMomentFpFloor(dataScale);
    this.v = 0;
    this.t = 0;
    this.lr = baseLr;
    this.lastSumSq = 0;
  }

  /** Update EMA / η from current grads. Does not touch parameters. */
  adaptLr(): { lr: number; sumSq: number } {
    let sumSq = 0;
    let count = 0;
    for (const p of this.params) {
      const g = p._grad;
      if (!g) continue;
      const flat = flatList(g.data);
      for (const x of flat) {
        sumSq += x * x;
        count += 1;
      }
    }
    const meanSq = count > 0 ? sumSq / count : 0;

    this.t += 1;
    this.v = this.beta2 * this.v + (1 - this.beta2) * meanSq;
    const vHat = this.v / (1 - Math.pow(this.beta2, this.t));
    const vUse = Math.max(vHat, meanSq, this.vFp);
    const denom = Math.sqrt(vUse) + Math.sqrt(this.vFp);
    this.lr = this.baseLr / denom;
    this.lastSumSq = sumSq;
    return { lr: this.lr, sumSq };
  }

  /** θ ← θ − scale·η·∇θ */
  applyScaled(scale: number): void {
    const step = -this.lr * scale;
    for (const p of this.params) {
      const g = p._grad;
      if (!g) continue;
      p._data = p.add(g.mul(step))._data;
    }
  }

  zero_grad(): void {
    for (const p of this.params) p.zero_grad?.();
  }
}

type Trainables = {
  U: AnyTensor;
  rawSigma: AnyTensor;
  V: AnyTensor;
  A: AnyTensor;
  optimizer: GlobalRmsSgd;
  m: number;
  n: number;
  k: number;
  step: number;
  device: DeviceKind;
};

/** Fully reduce (js-pytorch sum is one axis at a time). */
function fullSum(t: AnyTensor): AnyTensor {
  let s: AnyTensor = t;
  for (let guard = 0; guard < 8; guard++) {
    const flat = flatList(s.data);
    if (flat.length <= 1) return s;
    s = s.sum();
  }
  return s;
}

/**
 * Rank-k SVD factors with global-RMS SGD + QR retraction (js-pytorch).
 * Train on mean squared error (‖A−Â‖_F²/(mn)) so gradients stay O(1) as n grows;
 * charts still report Frobenius². After each step, thin-QR onto Stiefel.
 * Armijo backtracking on the retract rejects loss-increasing steps.
 * Sign/σ-order conventions are applied only when reading state for display.
 */
export class SvdGradTrainer {
  private t: Trainables | null = null;

  init(A: Matrix, rank: number, lr: number, device: DeviceKind): void {
    const m = A.rows;
    const n = A.cols;
    const k = Math.max(1, Math.min(rank, Math.min(m, n)));

    const U = torch.randn([m, k], true, false, device) as unknown as AnyTensor;
    const rawSigma = torch.randn([k], true, false, device) as unknown as AnyTensor;
    const V = torch.randn([n, k], true, false, device) as unknown as AnyTensor;
    scaleNestedInPlace(U.data, 0.3);
    scaleNestedInPlace(rawSigma.data, 0.3);
    scaleNestedInPlace(V.data, 0.3);
    retractInPlace(U, m, k);
    retractInPlace(V, n, k);

    const Aten = torch.tensor(toNested(A), false, device) as unknown as AnyTensor;
    const optimizer = new GlobalRmsSgd([U, rawSigma, V], lr, maxAbs(A));

    this.t = {
      U,
      rawSigma,
      V,
      A: Aten,
      optimizer,
      m,
      n,
      k,
      step: 0,
      device,
    };
  }

  setLr(lr: number): void {
    if (!this.t) return;
    this.t.optimizer.baseLr = lr;
  }

  get device(): DeviceKind {
    return this.t?.device ?? "cpu";
  }

  stepOnce(): GradState {
    const t = this.t;
    if (!t) throw new Error("trainer not initialized");

    const sigma = softplus(t.rawSigma, t.device);
    const Ahat = t.U.mul(sigma).matmul(t.V.transpose(0, 1));
    const mse = fullSum(Ahat.sub(t.A).pow(2)).mul(1 / (t.m * t.n));
    const L0 = flatList(mse.data)[0] ?? Infinity;

    mse.backward();
    const { lr, sumSq } = t.optimizer.adaptLr();

    const Amat = fromTensorData(t.A.data, t.m, t.n);
    const U0 = fromTensorData(t.U.data, t.m, t.k);
    const V0 = fromTensorData(t.V.data, t.n, t.k);
    const raw0 = flatList(t.rawSigma.data);
    const gU = t.U._grad
      ? fromTensorData(t.U._grad.data, t.m, t.k)
      : mat(t.m, t.k);
    const gV = t.V._grad
      ? fromTensorData(t.V._grad.data, t.n, t.k)
      : mat(t.n, t.k);
    const gRaw = t.rawSigma._grad
      ? flatList(t.rawSigma._grad.data)
      : raw0.map(() => 0);

    // ∇L · (−η g) = −η ‖g‖₂²  (Euclidean model; QR can invalidate it)
    const dirDeriv = -lr * sumSq;
    let alpha = 1;
    let accepted: {
      U: Matrix;
      V: Matrix;
      raw: number[];
    } | null = null;
    let bestDescent: {
      U: Matrix;
      V: Matrix;
      raw: number[];
      L: number;
    } | null = null;

    for (let bt = 0; bt < ARMIJO_MAX_BT; bt++) {
      const Utry = thinQ(addScaled(U0, gU, -alpha * lr));
      const Vtry = thinQ(addScaled(V0, gV, -alpha * lr));
      const rawTry = raw0.map((r, i) => r - alpha * lr * (gRaw[i] ?? 0));
      const Ltry = mseFactors(Amat, Utry, rawTry, Vtry);
      if (bestDescent === null || Ltry < bestDescent.L) {
        bestDescent = { U: Utry, V: Vtry, raw: rawTry, L: Ltry };
      }
      if (Ltry <= L0 + ARMIJO_C * alpha * dirDeriv) {
        accepted = { U: Utry, V: Vtry, raw: rawTry };
        break;
      }
      alpha *= 0.5;
    }

    // QR can make every Armijo trial fail even when some α still lowers the loss.
    // Prefer a plain descent step over freezing; only stay put if all trials increase L.
    if (!accepted && bestDescent && bestDescent.L < L0) {
      accepted = bestDescent;
    }

    if (accepted) {
      writeIntoNested(t.U.data, accepted.U);
      writeIntoNested(t.V.data, accepted.V);
      writeVectorIntoNested(t.rawSigma.data, accepted.raw);
    }
    t.optimizer.zero_grad();
    t.step += 1;

    return this.readState();
  }

  snapshot(_A: Matrix): GradState {
    if (!this.t) throw new Error("trainer not initialized");
    return this.readState();
  }

  /** Copy factors, apply display-only σ-sort / sign rule, report Frobenius². */
  private readState(): GradState {
    const t = this.t!;
    const Umat = fromTensorData(t.U.data, t.m, t.k);
    const Vmat = fromTensorData(t.V.data, t.n, t.k);
    const sigmaVals = flatList(softplus(t.rawSigma, t.device).data);
    const ordered = orderFactorsForDisplay(Umat, sigmaVals, Vmat);
    const AhatMat = reconstruct(ordered.U, ordered.sigma, ordered.V);
    const Amat = fromTensorData(t.A.data, t.m, t.n);
    const reconVal = frobeniusSq(sub(Amat, AhatMat));

    return {
      U: ordered.U,
      sigma: ordered.sigma,
      V: ordered.V,
      loss: { recon: reconVal },
      step: t.step,
      rank: t.k,
      device: t.device,
      lr: t.optimizer.lr,
    };
  }
}

/** Replace factor columns with thin-Q (Stiefel retraction). */
function retractInPlace(factor: AnyTensor, rows: number, cols: number): void {
  const m = fromTensorData(factor.data, rows, cols);
  writeIntoNested(factor.data, thinQ(m));
}

function addScaled(X: Matrix, G: Matrix, scale: number): Matrix {
  const out = copy(X);
  for (let i = 0; i < out.data.length; i++) {
    out.data[i]! += scale * (G.data[i] ?? 0);
  }
  return out;
}

function softplusNum(x: number): number {
  return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
}

function mseFactors(A: Matrix, U: Matrix, raw: number[], V: Matrix): number {
  const sigma = raw.map(softplusNum);
  const Ahat = reconstruct(U, sigma, V);
  return frobeniusSq(sub(A, Ahat)) / (A.rows * A.cols);
}

/** Write a 1-D vector into js-pytorch 1-D nested tensor storage. */
function writeVectorIntoNested(data: unknown, values: number[]): void {
  if (!Array.isArray(data)) return;
  if (data.length === values.length && typeof data[0] === "number") {
    for (let i = 0; i < values.length; i++) (data as number[])[i] = values[i]!;
    return;
  }
  // Sometimes stored as [[v0],[v1],…] or flat nested
  if (data.length === values.length && Array.isArray(data[0])) {
    for (let i = 0; i < values.length; i++) {
      const row = data[i] as number[];
      if (Array.isArray(row) && row.length >= 1) row[0] = values[i]!;
    }
    return;
  }
  for (let i = 0; i < Math.min(data.length, values.length); i++) {
    if (typeof data[i] === "number") (data as number[])[i] = values[i]!;
  }
}

/**
 * Sort columns by descending σ; flip so each U column’s largest-|entry| is ≥ 0.
 * Same rule as classicalSvd. Does not change Â. Display-only — does not mutate training tensors.
 */
function orderFactorsForDisplay(
  U: Matrix,
  sigma: number[],
  V: Matrix,
): { U: Matrix; sigma: number[]; V: Matrix } {
  const k = sigma.length;
  const order = Array.from({ length: k }, (_, i) => i).sort(
    (a, b) => sigma[b]! - sigma[a]!,
  );
  const Uo = mat(U.rows, k);
  const Vo = mat(V.rows, k);
  const so = new Array<number>(k);
  for (let j = 0; j < k; j++) {
    const src = order[j]!;
    so[j] = sigma[src]!;
    for (let i = 0; i < U.rows; i++) {
      Uo.data[i * k + j] = U.data[i * k + src]!;
    }
    for (let i = 0; i < V.rows; i++) {
      Vo.data[i * k + j] = V.data[i * k + src]!;
    }
  }
  for (let j = 0; j < k; j++) {
    let best = 0;
    let bestAbs = 0;
    for (let i = 0; i < Uo.rows; i++) {
      const v = Uo.data[i * k + j]!;
      if (Math.abs(v) > bestAbs) {
        bestAbs = Math.abs(v);
        best = v;
      }
    }
    if (best < 0) {
      for (let i = 0; i < Uo.rows; i++) Uo.data[i * k + j]! *= -1;
      for (let i = 0; i < Vo.rows; i++) Vo.data[i * k + j]! *= -1;
    }
  }
  return { U: Uo, sigma: so, V: Vo };
}

function softplus(raw: AnyTensor, device: DeviceKind): AnyTensor {
  const ones = torch.ones(raw.shape, false, device) as unknown as AnyTensor;
  return ones.add(raw.exp()).log();
}

function flatList(data: unknown): number[] {
  const out: number[] = [];
  const walk = (x: unknown) => {
    if (typeof x === "number") out.push(x);
    else if (Array.isArray(x)) x.forEach(walk);
  };
  walk(data);
  return out;
}

function scaleNestedInPlace(data: unknown, scale: number): void {
  if (!Array.isArray(data)) return;
  for (let i = 0; i < data.length; i++) {
    const el = data[i];
    if (typeof el === "number") (data as number[])[i] = el * scale;
    else scaleNestedInPlace(el, scale);
  }
}

/** Probe whether GPU.js / WebGL path works for a tiny matmul. */
export function probeGpu(): boolean {
  try {
    const a = torch.randn([2, 2], true, false, "gpu");
    const b = torch.randn([2, 2], false, false, "gpu");
    const c = a.matmul(b);
    c.pow(2).sum();
    return true;
  } catch {
    return false;
  }
}
