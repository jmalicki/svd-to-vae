import { torch } from "js-pytorch";
import {
  type Matrix,
  toNested,
  fromTensorData,
  frobeniusSq,
  reconstruct,
  sub,
  thinQ,
  writeIntoNested,
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
};

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

type Optimizer = { step: () => void; zero_grad: () => void; lr: number };

type Trainables = {
  U: AnyTensor;
  rawSigma: AnyTensor;
  V: AnyTensor;
  A: AnyTensor;
  optimizer: Optimizer;
  m: number;
  n: number;
  k: number;
  step: number;
  device: DeviceKind;
};

/**
 * Plain SGD — js-pytorch only ships Adam; moments fight QR retraction.
 * Update: θ ← θ − lr · ∇θ
 */
class Sgd {
  params: AnyTensor[];
  lr: number;

  constructor(params: AnyTensor[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  step(): void {
    for (const p of this.params) {
      const g = p._grad;
      if (!g) continue;
      p._data = p.add(g.mul(-this.lr))._data;
    }
  }

  zero_grad(): void {
    for (const p of this.params) p.zero_grad?.();
  }
}

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
 * Rank-k SVD factors with plain SGD + QR retraction (js-pytorch).
 * Minimize ||A − U diag(σ) Vᵀ||_F², then thin-QR U and V onto Stiefel each step.
 * σ = softplus(raw); Â = (U ⊙ σ) Vᵀ keeps σ in the graph.
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
    // Init: same sign rule as classical (σ sort kicks in once values differ).
    fixSignsInPlace(U, V, m, n, k);

    const Aten = torch.tensor(toNested(A), false, device) as unknown as AnyTensor;
    const optimizer = new Sgd([U, rawSigma, V], lr);

    this.t = { U, rawSigma, V, A: Aten, optimizer, m, n, k, step: 0, device };
  }

  setLr(lr: number): void {
    if (!this.t) return;
    this.t.optimizer.lr = lr;
  }

  get device(): DeviceKind {
    return this.t?.device ?? "cpu";
  }

  stepOnce(): GradState {
    const t = this.t;
    if (!t) throw new Error("trainer not initialized");

    const sigma = softplus(t.rawSigma, t.device);
    const Ahat = t.U.mul(sigma).matmul(t.V.transpose(0, 1));
    const recon = fullSum(Ahat.sub(t.A).pow(2));

    recon.backward();
    t.optimizer.step();
    t.optimizer.zero_grad();

    retractInPlace(t.U, t.m, t.k);
    retractInPlace(t.V, t.n, t.k);
    // Break sign/order ambiguity (L unchanged): σ descending; max-|U| entry ≥ 0 per column.
    fixSvdSignsAndOrderInPlace(t.U, t.V, t.rawSigma, t.m, t.n, t.k, t.device);
    t.step += 1;

    const Umat = fromTensorData(t.U.data, t.m, t.k);
    const Vmat = fromTensorData(t.V.data, t.n, t.k);
    const sigmaVals = flatList(softplus(t.rawSigma, t.device).data);
    const AhatMat = reconstruct(Umat, sigmaVals, Vmat);
    const Amat = fromTensorData(t.A.data, t.m, t.n);
    const reconVal = frobeniusSq(sub(Amat, AhatMat));

    return {
      U: Umat,
      sigma: sigmaVals,
      V: Vmat,
      loss: { recon: reconVal },
      step: t.step,
      rank: t.k,
      device: t.device,
    };
  }

  snapshot(A: Matrix): GradState {
    const t = this.t;
    if (!t) throw new Error("trainer not initialized");
    const sigmaVals = flatList(softplus(t.rawSigma, t.device).data);
    const Umat = fromTensorData(t.U.data, t.m, t.k);
    const Vmat = fromTensorData(t.V.data, t.n, t.k);
    const Ahat = reconstruct(Umat, sigmaVals, Vmat);
    const reconVal = frobeniusSq(sub(A, Ahat));
    return {
      U: Umat,
      sigma: sigmaVals,
      V: Vmat,
      loss: { recon: reconVal },
      step: t.step,
      rank: t.k,
      device: t.device,
    };
  }
}

/** Replace factor columns with thin-Q (Stiefel retraction). */
function retractInPlace(factor: AnyTensor, rows: number, cols: number): void {
  const m = fromTensorData(factor.data, rows, cols);
  writeIntoNested(factor.data, thinQ(m));
}

/**
 * Sort columns by descending σ; flip so each U column’s largest-|entry| is ≥ 0.
 * Same rule as classicalSvd. Does not change Â.
 */
function fixSvdSignsAndOrderInPlace(
  U: AnyTensor,
  V: AnyTensor,
  rawSigma: AnyTensor,
  m: number,
  n: number,
  k: number,
  device: DeviceKind,
): void {
  const sigma = flatList(softplus(rawSigma, device).data);
  const order = Array.from({ length: k }, (_, i) => i).sort(
    (a, b) => sigma[b] - sigma[a],
  );
  permuteColumnsInPlace(U.data, m, k, order);
  permuteColumnsInPlace(V.data, n, k, order);
  permuteVectorInPlace(rawSigma.data, order);
  fixSignsInPlace(U, V, m, n, k);
}

function fixSignsInPlace(
  U: AnyTensor,
  V: AnyTensor,
  m: number,
  n: number,
  k: number,
): void {
  const Umat = fromTensorData(U.data, m, k);
  const Vmat = fromTensorData(V.data, n, k);
  for (let j = 0; j < k; j++) {
    let best = 0;
    let bestAbs = 0;
    for (let i = 0; i < m; i++) {
      const v = Umat.data[i * k + j];
      if (Math.abs(v) > bestAbs) {
        bestAbs = Math.abs(v);
        best = v;
      }
    }
    if (best < 0) {
      for (let i = 0; i < m; i++) Umat.data[i * k + j] *= -1;
      for (let i = 0; i < n; i++) Vmat.data[i * k + j] *= -1;
    }
  }
  writeIntoNested(U.data, Umat);
  writeIntoNested(V.data, Vmat);
}

function permuteColumnsInPlace(
  data: unknown,
  rows: number,
  cols: number,
  order: number[],
): void {
  const src = fromTensorData(data, rows, cols);
  const out = {
    rows,
    cols,
    data: new Float64Array(rows * cols),
  };
  for (let j = 0; j < cols; j++) {
    const srcJ = order[j];
    for (let i = 0; i < rows; i++) {
      out.data[i * cols + j] = src.data[i * cols + srcJ];
    }
  }
  writeIntoNested(data, out);
}

function permuteVectorInPlace(data: unknown, order: number[]): void {
  if (!Array.isArray(data)) return;
  const src = (data as number[]).slice();
  for (let j = 0; j < order.length; j++) {
    (data as number[])[j] = src[order[j]];
  }
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
