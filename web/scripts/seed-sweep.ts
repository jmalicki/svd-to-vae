/**
 * Manual test: sweep seeds through the gradient-page trainer and report which
 * GD paths fail to converge to the truncated SVD (Eckart–Young) reconstruction.
 *
 * Each seed reproduces exactly what the page does with ?seed=… — same A, same
 * U/raw/V init — so any failing seed printed here can be replayed in the
 * browser. Failures are the point of the tool (finding bad basins), so it
 * always exits 0; read the report.
 *
 * Run (defaults match the page: n=5, k=3, lr=0.01):
 *   npm run test:seeds
 *   npm run test:seeds -- --seeds 100 --steps 3000 --n 6 --k 2 --lr 0.05
 *   npm run test:seeds -- --start 1                # deterministic seeds 1..N
 *
 * Diagnosis (hypothesis forensics on non-converged runs):
 *   npm run test:seeds -- --diagnose               # diagnose sweep failures
 *   npm run test:seeds -- --seedList 123,456       # diagnose specific seeds
 *   … --rescueSteps 20000 --rescueLr 0.05          # try to escape a stall
 *   … --rescueSteps 5000 --perturb 0.05            # kick the stuck point first
 *   … --initScale 1.0                              # init std-dev (default 0.3)
 *   … --warmStart                                  # spectral warm start (control)
 * Per-seed diagnosis reports: which singular pairs of A the run locked onto
 * (H1: wrong selection — predicted loss Σ excluded σ²), raw/softplus′ per
 * component (H2: frozen σ), and Armijo accept/α/‖g‖ over the last steps
 * (H3: optimizer freeze). Rescue results test H4 (slow escape).
 */
import { classicalSvd } from "../src/classicalSvd";
import {
  type Matrix,
  frobeniusSq,
  get,
  mulberry32,
  randomNormal,
  reconstruct,
  sub,
} from "../src/matrix";
import { SvdGradTrainer, type StepInfo } from "../src/svdGrad";

function numArg(name: string, dflt: number): number {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) return dflt;
  const v = Number(process.argv[i + 1]);
  return Number.isFinite(v) ? v : dflt;
}

function flagArg(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

function listArg(name: string): number[] {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) return [];
  return process.argv[i + 1]
    .split(",")
    .map((s) => Number(s.trim()) >>> 0)
    .filter((x) => Number.isFinite(x));
}

const N = numArg("n", 5);
const K = numArg("k", 3);
const LR = numArg("lr", 0.01);
const STEPS = numArg("steps", 2000);
const SEEDS = numArg("seeds", 40);
const START = numArg("start", NaN); // set for a deterministic sweep
const DIAGNOSE = flagArg("diagnose");
const SEED_LIST = listArg("seedList");
const RESCUE_STEPS = numArg("rescueSteps", 0);
const RESCUE_LR = numArg("rescueLr", NaN);
const PERTURB = numArg("perturb", 0); // noise std added to U/V before rescue
const INIT_SCALE = numArg("initScale", 0.3);
const WARM_START = flagArg("warmStart");
const WINDOW = numArg("window", 200);

// Converged runs sit ~1e-15 above the SVD reconstruction; bad critical points
// sit O(1) above it. Anything between is flagged as "undecided" (still moving?).
const CONVERGED_REL = 1e-9;
const FAILED_REL = 1e-4;

type Run = {
  seed: number;
  gap: number;
  rel: number;
  loss: number;
  floor: number;
  A: Matrix;
  trainer: SvdGradTrainer;
  window: StepInfo[];
};

function makeTrainer(A: Matrix, seed: number): SvdGradTrainer {
  const trainer = new SvdGradTrainer();
  const warmStart = WARM_START ? classicalSvd(A, K) : undefined;
  trainer.init(A, K, LR, "cpu", mulberry32(seed ^ 0x9e3779b9), {
    initScale: INIT_SCALE,
    warmStart,
  });
  return trainer;
}

function gapVsSvd(A: Matrix, trainer: SvdGradTrainer): { gap: number; loss: number; floor: number } {
  const svd = classicalSvd(A, K);
  const Asvd = reconstruct(svd.U, svd.sigma, svd.V);
  const state = trainer.snapshot(A);
  return {
    gap: frobeniusSq(sub(Asvd, reconstruct(state.U, state.sigma, state.V))),
    loss: state.loss.recon,
    floor: frobeniusSq(sub(A, Asvd)),
  };
}

function runSeed(seed: number): Run {
  const A = randomNormal(N, N, 1, mulberry32(seed));
  const trainer = makeTrainer(A, seed);

  const window: StepInfo[] = [];
  for (let i = 0; i < STEPS; i++) {
    const state = trainer.stepOnce();
    if (state.stepInfo && i >= STEPS - WINDOW) window.push(state.stepInfo);
  }

  const { gap, loss, floor } = gapVsSvd(A, trainer);
  return { seed, gap, rel: gap / frobeniusSq(A), loss, floor, A, trainer, window };
}

/** |X^T Y| for orthonormal-column matrices (rows: X cols, cols: Y cols). */
function absOverlap(X: Matrix, Y: Matrix): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < X.cols; i++) {
    const row: number[] = [];
    for (let j = 0; j < Y.cols; j++) {
      let dot = 0;
      for (let r = 0; r < X.rows; r++) dot += get(X, r, i) * get(Y, r, j);
      row.push(Math.abs(dot));
    }
    out.push(row);
  }
  return out;
}

function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function fmt(x: number): string {
  return x.toExponential(2);
}

function diagnose(r: Run): void {
  const { A, trainer, seed } = r;
  const full = classicalSvd(A, N); // all singular pairs of A
  const state = trainer.snapshot(A);

  console.log(`\n=== diagnose seed=${seed}  (replay: gradient.html?seed=${seed})`);
  console.log(
    `  final: L=${fmt(r.loss)}  floor=${fmt(r.floor)}  gap=${fmt(r.gap)}  rel=${fmt(r.rel)}`,
  );

  // --- H1: which singular pairs of A did the run lock onto?
  const overU = absOverlap(full.U, state.U); // n × k
  const overV = absOverlap(full.V, state.V);
  const picks: number[] = [];
  const pickQuality: number[] = [];
  for (let j = 0; j < state.U.cols; j++) {
    let bi = 0;
    let bv = -1;
    for (let i = 0; i < N; i++) {
      if (overU[i][j] > bv) {
        bv = overU[i][j];
        bi = i;
      }
    }
    picks.push(bi);
    pickQuality.push(bv);
  }
  const selection = [...new Set(picks)].sort((a, b) => a - b);
  const excluded = Array.from({ length: N }, (_, i) => i).filter((i) => !selection.includes(i));
  const predicted = excluded.reduce((s, i) => s + full.sigma[i] ** 2, 0);
  console.log(`  σ(A)         = [${full.sigma.map(fmt).join(", ")}]`);
  console.log(`  σ(GD sorted) = [${state.sigma.map(fmt).join(", ")}]`);
  console.log(
    `  H1 selection: GD column → A pair ${JSON.stringify(picks)} ` +
      `(U-overlap ${pickQuality.map(fmt).join(", ")})`,
  );
  console.log(
    `  H1 predicted loss for selection {${selection.join(",")}} = Σ excluded σ² = ${fmt(predicted)} ` +
      `vs actual ${fmt(r.loss)}  ${Math.abs(predicted - r.loss) < 0.01 * Math.max(predicted, 1e-12) ? "→ MATCH (wrong-pair critical point)" : "→ no match"}`,
  );
  if (picks.length !== selection.length) {
    console.log(`  H1 note: duplicate picks — some GD columns share one A pair`);
  }
  // V-side sanity: the right factors should agree on the same pair per column.
  const vAgrees = picks.every((p, j) => {
    let bi = 0;
    let bv = -1;
    for (let i = 0; i < N; i++) {
      if (overV[i][j] > bv) {
        bv = overV[i][j];
        bi = i;
      }
    }
    return bi === p;
  });
  if (!vAgrees) console.log(`  H1 note: U and V pick different A pairs for some column`);

  // --- H2: softplus freeze
  const raw = trainer.rawSigmaValues();
  const sig = raw.map((x) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0));
  const sp = raw.map(sigmoid);
  console.log(`  H2 raw       = [${raw.map(fmt).join(", ")}]`);
  console.log(`  H2 softplus′ = [${sp.map(fmt).join(", ")}]  σ(raw)=[${sig.map(fmt).join(", ")}]`);
  const frozen = raw.filter((x) => x < -5).length;
  if (frozen > 0) console.log(`  H2: ${frozen} component(s) with raw < −5 (σ≈0, gradient ≈ 0) → frozen`);

  // --- H3: Armijo behavior over the last ${WINDOW} steps
  const rejects = r.window.filter((w) => !w.accepted).length;
  const acceptedAlphas = r.window.filter((w) => w.accepted).map((w) => w.alpha);
  const meanAlpha =
    acceptedAlphas.length > 0
      ? acceptedAlphas.reduce((s, a) => s + a, 0) / acceptedAlphas.length
      : 0;
  const last = r.window.length > 0 ? r.window[r.window.length - 1] : undefined;
  const lastGrad = last?.gradNorm ?? NaN;
  const lastTangent = last?.tangentGradNorm ?? NaN;
  console.log(
    `  H3 last ${r.window.length} steps: rejects=${rejects}  mean α=${meanAlpha.toFixed(3)}  ` +
      `final ‖g‖=${fmt(lastGrad)}  tangent ‖g‖=${fmt(lastTangent)}`,
  );
  if (last) {
    console.log(
      `  H3 final-step trial ΔL (α=1,½,…): [${last.trialDeltas.map(fmt).join(", ")}]`,
    );
  }
  if (rejects === r.window.length) {
    if (lastTangent < 1e-6) {
      console.log(
        `  H3: all rejects but tangent gradient ≈ 0 → genuine constrained critical point (supports H1), not an optimizer bug`,
      );
    } else if (lastGrad > 1e-6) {
      console.log(
        `  H3: every recent step rejected with a sizable tangent gradient → OPTIMIZER FREEZE`,
      );
    }
  }

  // --- H4: rescue attempt
  if (RESCUE_STEPS > 0) {
    if (PERTURB > 0) trainer.perturb(PERTURB, mulberry32(seed ^ 0x5bd1e995));
    if (Number.isFinite(RESCUE_LR)) trainer.setLr(RESCUE_LR);
    for (let i = 0; i < RESCUE_STEPS; i++) trainer.stepOnce();
    const after = gapVsSvd(A, trainer);
    const rel = after.gap / frobeniusSq(A);
    console.log(
      `  H4 rescue (+${RESCUE_STEPS} steps` +
        `${Number.isFinite(RESCUE_LR) ? `, lr=${RESCUE_LR}` : ""}` +
        `${PERTURB > 0 ? `, perturb=${PERTURB}` : ""}): ` +
        `gap ${fmt(r.gap)} → ${fmt(after.gap)}  ${rel <= CONVERGED_REL ? "→ ESCAPED" : rel >= FAILED_REL ? "→ still stuck" : "→ moving"}`,
    );
  }
}

function main(): void {
  const explicit = SEED_LIST.length > 0;
  const seeds: number[] = explicit
    ? SEED_LIST
    : Array.from({ length: SEEDS }, (_, i) =>
        Number.isFinite(START) ? (START + i) >>> 0 : (Math.random() * 0x100000000) >>> 0,
      );

  console.log(
    `seed sweep: n=${N} k=${K} lr=${LR} steps=${STEPS} initScale=${INIT_SCALE}` +
      `${WARM_START ? " warmStart" : ""} · ${seeds.length} seeds ` +
      `(${explicit ? "explicit list" : Number.isFinite(START) ? `deterministic from ${START}` : "random"})`,
  );

  const failed: Run[] = [];
  const undecided: Run[] = [];
  let converged = 0;

  for (const seed of seeds) {
    const r = runSeed(seed);
    if (r.rel <= CONVERGED_REL) {
      converged += 1;
      if (explicit) console.log(`  ok    seed=${r.seed}  gap=${fmt(r.gap)}`);
    } else if (r.rel >= FAILED_REL) {
      failed.push(r);
      console.log(
        `  FAIL  seed=${r.seed}  ‖Â_svd−Â_gd‖²=${fmt(r.gap)} ` +
          `(rel ${fmt(r.rel)})  L=${fmt(r.loss)} vs floor ${fmt(r.floor)}`,
      );
    } else {
      undecided.push(r);
      console.log(
        `  ....  seed=${r.seed}  gap=${fmt(r.gap)} — neither converged nor stuck; try more --steps`,
      );
    }
  }

  console.log(
    `\n${converged}/${seeds.length} converged to the truncated SVD · ` +
      `${failed.length} stuck at a worse critical point · ${undecided.length} undecided`,
  );

  if (DIAGNOSE || explicit) {
    for (const r of [...failed, ...undecided]) diagnose(r);
  } else if (failed.length > 0) {
    console.log(`\nreplay a stuck run in the browser:`);
    for (const r of failed) console.log(`  gradient.html?seed=${r.seed}`);
    console.log(`or diagnose here: npm run test:seeds -- --seedList ${failed.map((r) => r.seed).join(",")} --diagnose`);
  }
}

main();
