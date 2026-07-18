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
 *   npm run test:seeds -- --start 1     # deterministic sweep: seeds 1..N
 */
import { classicalSvd } from "../src/classicalSvd";
import {
  frobeniusSq,
  mulberry32,
  randomNormal,
  reconstruct,
  sub,
} from "../src/matrix";
import { SvdGradTrainer } from "../src/svdGrad";

function numArg(name: string, dflt: number): number {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) return dflt;
  const v = Number(process.argv[i + 1]);
  return Number.isFinite(v) ? v : dflt;
}

const N = numArg("n", 5);
const K = numArg("k", 3);
const LR = numArg("lr", 0.01);
const STEPS = numArg("steps", 2000);
const SEEDS = numArg("seeds", 40);
const START = numArg("start", NaN); // set for a deterministic sweep

// Converged runs sit ~1e-15 above the SVD reconstruction; bad critical points
// sit O(1) above it. Anything between is flagged as "undecided" (still moving?).
const CONVERGED_REL = 1e-9;
const FAILED_REL = 1e-4;

type Run = { seed: number; gap: number; rel: number; loss: number; floor: number };

function runSeed(seed: number): Run {
  const A = randomNormal(N, N, 1, mulberry32(seed));
  const svd = classicalSvd(A, K);
  const Asvd = reconstruct(svd.U, svd.sigma, svd.V);
  const floor = frobeniusSq(sub(A, Asvd));

  const trainer = new SvdGradTrainer();
  trainer.init(A, K, LR, "cpu", mulberry32(seed ^ 0x9e3779b9));
  let state = trainer.snapshot(A);
  for (let i = 0; i < STEPS; i++) state = trainer.stepOnce();

  const gap = frobeniusSq(sub(Asvd, reconstruct(state.U, state.sigma, state.V)));
  return { seed, gap, rel: gap / frobeniusSq(A), loss: state.loss.recon, floor };
}

function main(): void {
  const seeds: number[] = [];
  for (let i = 0; i < SEEDS; i++) {
    seeds.push(
      Number.isFinite(START) ? (START + i) >>> 0 : (Math.random() * 0x100000000) >>> 0,
    );
  }

  console.log(
    `seed sweep: n=${N} k=${K} lr=${LR} steps=${STEPS} · ${SEEDS} seeds ` +
      `(${Number.isFinite(START) ? `deterministic from ${START}` : "random"})`,
  );

  const failed: Run[] = [];
  const undecided: Run[] = [];
  let converged = 0;

  for (const seed of seeds) {
    const r = runSeed(seed);
    if (r.rel <= CONVERGED_REL) {
      converged += 1;
    } else if (r.rel >= FAILED_REL) {
      failed.push(r);
      console.log(
        `  FAIL  seed=${r.seed}  ‖Â_svd−Â_gd‖²=${r.gap.toExponential(3)} ` +
          `(rel ${r.rel.toExponential(2)})  L=${r.loss.toExponential(3)} vs floor ${r.floor.toExponential(3)}`,
      );
    } else {
      undecided.push(r);
      console.log(
        `  ....  seed=${r.seed}  gap=${r.gap.toExponential(3)} — neither converged nor stuck; try more --steps`,
      );
    }
  }

  console.log(
    `\n${converged}/${SEEDS} converged to the truncated SVD · ` +
      `${failed.length} stuck at a worse critical point · ${undecided.length} undecided`,
  );
  if (failed.length > 0) {
    console.log(`\nreplay a stuck run in the browser:`);
    for (const r of failed) console.log(`  gradient.html?seed=${r.seed}`);
  }
}

main();
