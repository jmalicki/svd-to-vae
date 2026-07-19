/**
 * Build-time ablation: the Armijo-freeze failure mode and the tangent-projection
 * fix, measured on the exact run the gradient page produces for ?seed=1604623524
 * (n=5, k=3, lr=0.01 — same A, same U/raw/V init).
 *
 * Measures the full 2×2×2 cube {v† floor} × {Armijo} × {tangent projection} with
 * the shared loop in scripts/ablation-lib.mjs, plots the floor-on plane
 * (projection × Armijo) to public/freeze-tangent.svg, and exits nonzero if any
 * cell stops behaving the way the gradient page's appendix claims. Measured
 * behavior on this seed (floor on or off — the floor is not the actor here):
 *   raw + Armijo        → freeze: every late step rejected, gap parked at
 *                         rel ≈ 0.47 forever. The bad direction and the veto interlock.
 *   raw, no Armijo      → escapes the freeze (uphill steps are allowed through the
 *                         bad region) and touches machine precision near step 3300 —
 *                         then drifts off and buzzes around rel ≈ 1e-3: with no veto,
 *                         once the gap hits the noise floor the RMS-normalized steps
 *                         keep kicking the iterate off the solution. A 300-seed sweep
 *                         shows the same tail on every no-Armijo run.
 *   projected, no Armijo→ same tail failure with a sound direction: dives straight
 *                         to ~1e-16 (no excursion), then drifts up to rel ≈ 6e-4.
 *   projected + Armijo  → the live demo: machine precision, and *stays*.
 * Moral: projection fixes the direction; Armijo is what makes convergence permanent.
 *
 * Run: node scripts/gen-freeze-plot.mjs  (--probe to print all 8 cells)
 */
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  frobeniusSq,
  pageInit,
  pageMatrix,
  renderSvg,
  train,
  truncSvd,
} from "./ablation-lib.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, "../public/freeze-tangent.svg");

// Page defaults; the seed is a real formerly-frozen run, replayable in the
// browser as gradient.html?seed=1604623524.
// 6000 steps, not 2000: the no-Armijo failure is a *tail* failure. Both
// no-Armijo variants touch the SVD and only get ejected later (the floored
// step size balloons once gradients hit the noise floor and nothing vetoes
// the kick); a 2000-step window would catch them at a flattering moment.
const N = 5;
const K = 3;
const STEPS = 6000;
const BASE_LR = 0.01;
const SEED = 1604623524;

const PROBE = process.argv.includes("--probe");

function fmt(x) {
  return x.toExponential(2);
}

function main() {
  const A = pageMatrix(SEED, N);
  const Asvd = truncSvd(A, K);
  const init = pageInit(SEED, N, K);
  const normA = frobeniusSq(A);
  const cfg = { k: K, steps: STEPS, baseLr: BASE_LR };

  const cells = {};
  for (const project of [false, true]) {
    for (const floor of [true, false]) {
      for (const armijo of [true, false]) {
        const name = `${project ? "proj" : "raw"}/${floor ? "floor" : "nofloor"}/${armijo ? "armijo" : "noarmijo"}`;
        cells[name] = train(A, Asvd, { ...cfg, project, floor, armijo }, init);
      }
    }
  }

  if (PROBE) {
    for (const [name, r] of Object.entries(cells)) {
      console.log(
        `${name.padEnd(26)} finalRel=${fmt(r.finalGap / normA)} minRel=${fmt(Math.min(...r.series) / normA)} ` +
          `tailRejects=${r.tailRejects}/${r.tailWindow}`,
      );
    }
    return;
  }

  const rel = (r) => r.finalGap / normA;
  const minRel = (r) => Math.min(...r.series) / normA;

  // --- assertions: each documented (measured) behavior must keep reproducing ---
  const checks = [
    // The freeze itself (the old demo): direction is ascent, Armijo vetoes forever.
    ["raw/floor/armijo parked O(1) above the SVD gap", rel(cells["raw/floor/armijo"]) > 1e-2],
    [
      "raw/floor/armijo rejects every late step",
      cells["raw/floor/armijo"].tailRejects === cells["raw/floor/armijo"].tailWindow,
    ],
    // Same freeze without the floor: the floor is not the actor in this failure.
    ["raw/nofloor/armijo equally frozen", rel(cells["raw/nofloor/armijo"]) > 1e-2],
    // No veto ⇒ no freeze: uphill steps are allowed through the bad region and
    // the run touches the SVD mid-flight…
    ["raw/floor/noarmijo escapes the freeze (touches the SVD)", minRel(cells["raw/floor/noarmijo"]) < 1e-9],
    // …but with no veto it cannot *stay*: unvetoed RMS steps eject the iterate
    // and it ends buzzing around rel ≈ 1e-3.
    ["raw/floor/noarmijo does not stay converged", rel(cells["raw/floor/noarmijo"]) > 1e-6],
    // The fix: same seed, same init, projected gradient converges under Armijo
    // and remains parked (Armijo vetoes the ejection kicks near the solution).
    ["proj/floor/armijo converges to machine precision and stays", rel(cells["proj/floor/armijo"]) < 1e-9],
    ["proj/nofloor/armijo converges too", rel(cells["proj/nofloor/armijo"]) < 1e-9],
    // Sound direction, no veto: dives straight to the SVD (no excursion), then
    // suffers the same tail ejection. Armijo's job is permanence, not descent.
    ["proj/floor/noarmijo touches the SVD", minRel(cells["proj/floor/noarmijo"]) < 1e-9],
    ["proj/floor/noarmijo does not stay converged", rel(cells["proj/floor/noarmijo"]) > 1e-6],
  ];
  const failures = checks.filter(([, ok]) => !ok).map(([what]) => what);
  if (failures.length > 0) {
    for (const [name, r] of Object.entries(cells)) {
      console.error(
        `${name.padEnd(26)} finalGap=${fmt(r.finalGap)} rel=${fmt(rel(r))} ` +
          `tailRejects=${r.tailRejects}/${r.tailWindow}`,
      );
    }
    throw new Error(`freeze ablation drifted from documented behavior:\n  - ${failures.join("\n  - ")}`);
  }

  mkdirSync(path.dirname(OUT), { recursive: true });
  // Decimate for file size: the plot is ~480px wide, so keep ~750 buckets but
  // preserve each bucket's min and max so log-scale dives and the buzzing
  // no-Armijo tail keep their true envelope.
  const thin = (series) => {
    const buckets = 750;
    if (series.length <= 2 * buckets) return series;
    const out = [];
    const per = series.length / buckets;
    for (let b = 0; b < buckets; b++) {
      const from = Math.floor(b * per);
      const to = Math.min(series.length, Math.floor((b + 1) * per));
      let lo = Infinity;
      let hi = -Infinity;
      let loAt = from;
      let hiAt = from;
      for (let i = from; i < to; i++) {
        if (series[i] < lo) { lo = series[i]; loAt = i; }
        if (series[i] > hi) { hi = series[i]; hiAt = i; }
      }
      out.push(...(loAt <= hiAt ? [lo, hi] : [hi, lo]));
    }
    return out;
  };
  // Figure: the floor-on plane. Array order (reversed) sets the legend;
  // drawOrder sets z-stacking (dashed curves above the solids they overlap).
  writeFileSync(
    OUT,
    renderSvg(
      [
        { series: thin(cells["raw/floor/armijo"].series), color: "#E69F00", width: 2.25, label: "raw (freeze)", drawOrder: 1 },
        { series: thin(cells["raw/floor/noarmijo"].series), color: "#D55E00", width: 1.75, dash: "5 3", label: "raw, no Armijo", drawOrder: 2 },
        { series: thin(cells["proj/floor/noarmijo"].series), color: "#009E73", width: 1.75, dash: "5 3", label: "proj, no Armijo", drawOrder: 3 },
        { series: thin(cells["proj/floor/armijo"].series), color: "#0072B2", width: 2.25, label: "proj+Armijo", drawOrder: 0 },
      ],
      {
        ariaLabel:
          "Measured ‖Â_svd − Â_gd‖_F² on page seed 1604623524: raw-gradient runs freeze above the SVD gap, tangent-projected runs converge",
        footer: `n=${N} k=${K} steps=${STEPS} lr=${BASE_LR} seed=${SEED} (gradient.html?seed=${SEED}) · v† floor on · generated by scripts/gen-freeze-plot.mjs`,
      },
    ),
  );

  console.log(`wrote ${OUT}\n  metric: ‖Â_svd − Â_gd‖_F² · page seed ${SEED}`);
  for (const [name, r] of Object.entries(cells)) {
    console.log(
      `  ${name.padEnd(24)} finalGap=${fmt(r.finalGap)} rel=${fmt(rel(r))} ` +
        `tailRejects=${r.tailRejects}/${r.tailWindow}`,
    );
  }
}

main();
