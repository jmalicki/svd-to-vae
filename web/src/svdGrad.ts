import { torch } from "js-pytorch";
import {
  type Matrix,
  toNested,
  fromTensorData,
  identity,
  frobeniusSq,
  reconstruct,
  sub,
  matmul,
  transpose,
} from "./matrix";

export type DeviceKind = "cpu" | "gpu";

export type LossParts = {
  total: number;
  recon: number;
  ortho: number;
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
};

type Trainables = {
  U: AnyTensor;
  rawSigma: AnyTensor;
  V: AnyTensor;
  A: AnyTensor;
  optimizer: { step: () => void; zero_grad: () => void; lr: number };
  m: number;
  n: number;
  k: number;
  lambda: number;
  step: number;
  device: DeviceKind;
};

function nestedEye(k: number): number[][] {
  return Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)),
  );
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
 * Soft-constrained rank-k SVD objective with Adam (js-pytorch).
 * σ = softplus(raw); Â = (U ⊙ σ) Vᵀ keeps σ in the graph.
 */
export class SvdGradTrainer {
  private t: Trainables | null = null;

  init(A: Matrix, rank: number, lambda: number, lr: number, device: DeviceKind): void {
    const m = A.rows;
    const n = A.cols;
    const k = Math.max(1, Math.min(rank, Math.min(m, n)));

    const U = torch.randn([m, k], true, false, device) as unknown as AnyTensor;
    const rawSigma = torch.randn([k], true, false, device) as unknown as AnyTensor;
    const V = torch.randn([n, k], true, false, device) as unknown as AnyTensor;
    scaleNestedInPlace(U.data, 0.3);
    scaleNestedInPlace(rawSigma.data, 0.3);
    scaleNestedInPlace(V.data, 0.3);

    const Aten = torch.tensor(toNested(A), false, device) as unknown as AnyTensor;
    const optimizer = new torch.optim.Adam(
      [U, rawSigma, V] as never[],
      lr,
      0,
    );

    this.t = { U, rawSigma, V, A: Aten, optimizer, m, n, k, lambda, step: 0, device };
  }

  setHyperparams(lambda: number, lr: number): void {
    if (!this.t) return;
    this.t.lambda = lambda;
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

    const I = torch.tensor(nestedEye(t.k), false, t.device) as unknown as AnyTensor;
    const orthoU = fullSum(t.U.transpose(0, 1).matmul(t.U).sub(I).pow(2));
    const orthoV = fullSum(t.V.transpose(0, 1).matmul(t.V).sub(I).pow(2));
    const ortho = orthoU.add(orthoV);
    const loss = recon.add(ortho.mul(t.lambda));

    const reconVal = scalarValue(recon);
    const orthoVal = scalarValue(ortho);
    const totalVal = reconVal + t.lambda * orthoVal;

    loss.backward();
    t.optimizer.step();
    t.optimizer.zero_grad();
    t.step += 1;

    return this.pack(totalVal, reconVal, orthoVal);
  }

  snapshot(A: Matrix): GradState {
    const t = this.t;
    if (!t) throw new Error("trainer not initialized");
    const sigmaVals = flatList(softplus(t.rawSigma, t.device).data);
    const Umat = fromTensorData(t.U.data, t.m, t.k);
    const Vmat = fromTensorData(t.V.data, t.n, t.k);
    const Ahat = reconstruct(Umat, sigmaVals, Vmat);
    const reconVal = frobeniusSq(sub(A, Ahat));
    const orthoVal =
      frobeniusSq(sub(matmul(transpose(Umat), Umat), identity(t.k))) +
      frobeniusSq(sub(matmul(transpose(Vmat), Vmat), identity(t.k)));
    return {
      U: Umat,
      sigma: sigmaVals,
      V: Vmat,
      loss: { total: reconVal + t.lambda * orthoVal, recon: reconVal, ortho: orthoVal },
      step: t.step,
      rank: t.k,
      device: t.device,
    };
  }

  private pack(total: number, recon: number, ortho: number): GradState {
    const t = this.t!;
    return {
      U: fromTensorData(t.U.data, t.m, t.k),
      sigma: flatList(softplus(t.rawSigma, t.device).data),
      V: fromTensorData(t.V.data, t.n, t.k),
      loss: { total, recon, ortho },
      step: t.step,
      rank: t.k,
      device: t.device,
    };
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

function scalarValue(t: AnyTensor): number {
  return flatList(t.data).reduce((a, b) => a + b, 0);
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
