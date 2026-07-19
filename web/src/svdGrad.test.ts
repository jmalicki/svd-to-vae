import { describe, expect, it } from "vitest";
import { classicalSvd } from "./classicalSvd";
import {
  frobeniusSq,
  get,
  mat,
  matmul,
  mulberry32,
  randomNormal,
  reconstruct,
  sub,
  thinQ,
  transpose,
} from "./matrix";
import { SvdGradTrainer } from "./svdGrad";

function randnMat(rows: number, cols: number, rand: () => number, scale = 1) {
  const A = mat(rows, cols);
  for (let i = 0; i < A.data.length; i++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    A.data[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return A;
}

/** Rank-k matrix plus tiny noise — EY target is unambiguous. */
function lowRankA(n: number, k: number, seed: number) {
  const rand = mulberry32(seed);
  const U = thinQ(randnMat(n, k, rand, 1));
  const V = thinQ(randnMat(n, k, rand, 1));
  const sigma = Array.from({ length: k }, (_, j) => 3 - 0.7 * j);
  const A = reconstruct(U, sigma, V);
  const noise = randnMat(n, n, rand, 0.02);
  for (let i = 0; i < A.data.length; i++) A.data[i]! += noise.data[i]!;
  return A;
}

function expectOrtho(X: ReturnType<typeof mat>, digits = 8) {
  const XtX = matmul(transpose(X), X);
  for (let i = 0; i < X.cols; i++) {
    for (let j = 0; j < X.cols; j++) {
      expect(get(XtX, i, j)).toBeCloseTo(i === j ? 1 : 0, digits);
    }
  }
}

function countRejects(trainer: SvdGradTrainer, steps: number) {
  let rejects = 0;
  let finalLoss = 0;
  let sigmas: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t = (
      trainer as unknown as {
        t: { U: { data: unknown }; rawSigma: { data: unknown } };
      }
    ).t;
    const before = JSON.stringify(t.U.data) + JSON.stringify(t.rawSigma.data);
    const s = trainer.stepOnce();
    const after = JSON.stringify(t.U.data) + JSON.stringify(t.rawSigma.data);
    if (before === after) rejects += 1;
    finalLoss = s.loss.recon;
    sigmas = s.sigma;
  }
  return { rejects, finalLoss, sigmas };
}

describe("SvdGradTrainer", () => {
  it("keeps singular values strictly positive (softplus)", () => {
    const A = lowRankA(5, 3, 1);
    const trainer = new SvdGradTrainer();
    trainer.init(A, 3, 0.05, "cpu");
    const { sigmas } = countRejects(trainer, 40);
    expect(sigmas.length).toBe(3);
    for (const s of sigmas) {
      expect(s).toBeGreaterThan(0);
      expect(Number.isFinite(s)).toBe(true);
    }
  });

  it("optimizes: loss drops, early steps are accepted, and Â nears truncated SVD", () => {
    // Several matrices — catches Armijo freeze (rejecting every trial while loss is still large).
    for (const seed of [7, 11, 19, 23]) {
      const A = lowRankA(6, 2, seed);
      const k = 2;
      const svd = classicalSvd(A, k);
      const Asvd = reconstruct(svd.U, svd.sigma, svd.V);

      let ok = false;
      for (let attempt = 0; attempt < 5 && !ok; attempt++) {
        const trainer = new SvdGradTrainer();
        trainer.init(A, k, 0.05, "cpu");
        const start = trainer.snapshot(A).loss.recon;

        // Near a minimum, rejecting loss-increasing steps is correct — only police
        // freezes while still far from the truncated SVD.
        const early = countRejects(trainer, 120);
        expect(early.rejects).toBeLessThan(25);
        expect(early.finalLoss).toBeLessThan(start * 0.5);

        for (let i = 0; i < 280; i++) trainer.stepOnce();
        const gd = trainer.snapshot(A);
        const gap = frobeniusSq(sub(Asvd, reconstruct(gd.U, gd.sigma, gd.V)));
        expect(gd.loss.recon).toBeLessThan(start * 0.2);
        if (gap < 0.05) ok = true;
      }
      expect(ok).toBe(true);
    }
  });

  it("reproduces a run bit-for-bit from a seeded init", () => {
    const A = lowRankA(6, 2, 5);
    const t1 = new SvdGradTrainer();
    const t2 = new SvdGradTrainer();
    t1.init(A, 2, 0.05, "cpu", mulberry32(123));
    t2.init(A, 2, 0.05, "cpu", mulberry32(123));
    for (let i = 0; i < 25; i++) {
      t1.stepOnce();
      t2.stepOnce();
    }
    const s1 = t1.snapshot(A);
    const s2 = t2.snapshot(A);
    expect(Array.from(s1.U.data)).toEqual(Array.from(s2.U.data));
    expect(Array.from(s1.V.data)).toEqual(Array.from(s2.V.data));
    expect(s1.sigma).toEqual(s2.sigma);
    expect(s1.loss.recon).toBe(s2.loss.recon);
  });

  it("does not freeze on seeds that stalled without tangent projection", () => {
    // Regression for the ~8% Armijo-freeze failure mode: with the raw Euclidean
    // gradient fed through the QR retraction, these page seeds (n=5, k=3,
    // lr=0.01 — exact gradient.html?seed=… replays) rejected every step from
    // ~step 200 on and parked at 20–50% relative gap above the Eckart–Young
    // floor. Projecting U/V grads onto the Stiefel tangent space fixes them.
    for (const seed of [1604623524, 3401442139]) {
      const A = randomNormal(5, 5, 1, mulberry32(seed));
      const svd = classicalSvd(A, 3);
      const Asvd = reconstruct(svd.U, svd.sigma, svd.V);

      const trainer = new SvdGradTrainer();
      trainer.init(A, 3, 0.01, "cpu", mulberry32(seed ^ 0x9e3779b9));
      for (let i = 0; i < 2000; i++) trainer.stepOnce();

      const gd = trainer.snapshot(A);
      const gap = frobeniusSq(sub(Asvd, reconstruct(gd.U, gd.sigma, gd.V)));
      expect(gap / frobeniusSq(A)).toBeLessThan(1e-9);
    }
  });

  it("preserves orthonormal U and V after QR-retracted steps", () => {
    const A = lowRankA(5, 3, 3);
    const trainer = new SvdGradTrainer();
    trainer.init(A, 3, 0.05, "cpu");
    for (let i = 0; i < 30; i++) trainer.stepOnce();
    const s = trainer.snapshot(A);
    expectOrtho(s.U);
    expectOrtho(s.V);
  });
});
