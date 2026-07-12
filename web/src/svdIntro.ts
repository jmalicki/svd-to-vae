import "./style.css";
import { chapterNav } from "./chapterNav";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import {
  applyLeft,
  applyMat2,
  applyRight,
  columnOf,
  decomposeAlongNormal,
  demoMatrix2,
  householderAimColumn,
  householderAimToE1,
  householderFromNormal,
  normalFromLineAngle,
  reflectAcrossNormal,
  rightGivensZeroSuperdiag,
} from "./householder2d";
import { prepareHiDpiCanvas } from "./hiDpiCanvas";
import { frameFromMatrix } from "./svdGeometry2d";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
  get,
  fromNested,
  copy,
} from "./matrix";
import { drawHeatmap, drawSigmaBars } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const INK = "#1a1a1a";
const MUTED = "#9a9a9a";
const ACCENT = "#0072B2";
const ACCENT2 = "#E69F00";
const MIRROR = "#D55E00";

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    ${chapterNav({
      current: 2,
      prev: { href: "./", label: "← Matrix" },
      next: { href: "./truncate.html", label: "Next →" },
    })}
    <h1>Reflections, zeros, and stretches</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-to-vae" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      Often you want a vector — or a matrix column — lined up with a coordinate axis:
      one nonzero entry, the rest zero, length unchanged. A reflection can do that in one step.
      This page builds that operator from scratch, then uses it to clear zeros under a pivot
      and read the stretches that remain.
    </p>
  </header>

  <section class="theory" aria-label="Build a reflection">
    <h2>Building a reflection</h2>
    <p>
      Chapter 1’s rotations keep every length the same and never flip the plane.
      A <strong>reflection</strong> is the other elementary length-preserving map: it keeps
      lengths, but sends each point to its twin on the opposite side of a chosen line
      (the <strong>mirror</strong>). Through the origin, that line is fixed by a single
      direction — any unit vector $n$ perpendicular to the mirror, called the
      <strong>normal</strong>. Flip $n$ to $-n$ and you describe the same mirror; what matters
      is the line $n$ is orthogonal to.
    </p>
    <p>
      Once $n$ is chosen, reflecting a vector $x$ is a three-line geometric story.
      First draw the mirror and its normal. The figure below uses the same slider for all
      three panels: turn the orange mirror and the green normal $n$ turns with it.
    </p>
    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Mirror angle (sets $n$) <strong id="buildAngVal">35°</strong></span>
          <input id="buildAng" type="range" min="-90" max="90" step="1" value="35" />
        </label>
        <p class="help">One control for the whole construction below.</p>
      </div>
    </div>
    <p class="formula" id="buildFormula" aria-live="polite"></p>

    <div class="panel">
      <h2>1 · Mirror and normal</h2>
      <p>
        The orange band is the mirror line through the origin. The green arrow is a unit
        normal $n$ — length $1$, at right angles to the mirror. Choosing the mirror or
        choosing $n$ is the same decision: the reflection is completely fixed once you
        pick either one.
      </p>
      <canvas id="build1" width="300" height="300" aria-label="Mirror and normal"></canvas>
      <p class="hint">Orange = mirror. Green = $n$. Gray = coordinate axes.</p>
    </div>

    <div class="panel">
      <h2>2 · Split $x$ into parallel and normal pieces</h2>
      <p>
        Take any probe vector $x$ (blue). Drop a perpendicular from its tip onto the mirror.
        The foot of that perpendicular is the part of $x$ that already lies in the mirror
        direction — call it $x_{\\|}$. The leftover segment, from that foot out to the tip of
        $x$, points exactly along $\\pm n$. Its signed length is the scalar $n^{\\top}x$, so the
        leftover vector is
      </p>
      <div class="math">
        $$x_{n} = (n^{\\top}x)\\,n.$$
      </div>
      <p>
        Then $x = x_{\\|} + x_{n}$: the blue arrow is the vector sum of the purple
        (parallel) piece and the green (normal) piece in the figure.
      </p>
      <canvas id="build2" width="300" height="300" aria-label="Decompose x"></canvas>
      <p class="hint">Blue = $x$. Purple = $x_{\\|}$ along the mirror. Green stub = $x_n$ along $n$.</p>
    </div>

    <div class="panel">
      <h2>3 · Keep the parallel part, reverse the normal part</h2>
      <p>
        Reflecting across the mirror means: leave $x_{\\|}$ alone, and send $x_{n}$ to $-x_{n}$.
        The image is therefore $x_{\\|} - x_{n}$. Substituting $x_{n} = (n^{\\top}x)\\,n$ and
        $x_{\\|} = x - x_{n}$ gives
      </p>
      <div class="math">
        $$Hx = x_{\\|} - x_{n} = x - 2(n^{\\top}x)\\,n.$$
      </div>
      <p>
        Geometrically you subtract the normal piece twice: once to cancel it, once to go the
        same distance to the other side of the mirror. The orange arrow $Hx$ is the reflected
        tip; the dashed segment shows the jump from $x$ to $Hx$. Lengths match:
        $\\|Hx\\| = \\|x\\|$.
      </p>
      <canvas id="build3" width="300" height="300" aria-label="Reflect x to Hx"></canvas>
      <p class="hint">Blue = $x$. Orange = $Hx$. Dashed = the normal flip.</p>
    </div>

    <div class="panel">
      <h2>4 · Same rule as a matrix</h2>
      <p>
        The map $x \\mapsto x - 2(n^{\\top}x)\\,n$ is linear in $x$. Factoring gives the
        familiar Householder matrix
      </p>
      <div class="math">
        $$H = I - 2nn^{\\top}.$$
      </div>
      <p>
        Because reflecting twice is the identity, $H^{2} = I$. Because reflections preserve
        lengths and angles (up to orientation), $H$ is orthogonal: $H^{\\top}H = I$. The flip
        of orientation shows up as $\\det H = -1$ — unlike the rotations from chapter 1, which
        had determinant $+1$. From here on, “apply a reflection” means multiply by such an $H$.
      </p>
    </div>
  </section>

  <section class="demo-block" aria-label="Check reflection">
    <h2>Check: choose $n$ so $Hx$ hits a target</h2>
    <p class="demo-intro">
      Same $H = I - 2nn^{\\top}$. Fix $x$ (solid orange) and a target $y$ (dashed blue).
      Find the mirror — equivalently find $n$ — such that $Hx = y$.
      When $\\|Hx - y\\| = 0$ you have the right reflection; $\\|Hx\\| = \\|x\\|$ the whole time.
    </p>
    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Mirror angle (sets $n$) <strong id="mirAngVal">10°</strong></span>
          <input id="mirAng" type="range" min="-90" max="90" step="1" value="10" />
        </label>
        <p class="help">Right panel shows $Hx$. Match it to the dashed target $y$.</p>
      </div>
    </div>
    <div class="hunt-meter" aria-hidden="true">
      <span class="hunt-meter-label">$\\|Hx - y\\|$</span>
      <div class="hunt-meter-track">
        <div class="hunt-meter-fill" id="mirMeterFill"></div>
      </div>
      <strong class="hunt-meter-val" id="mirMeterVal">—</strong>
    </div>
    <div class="grid-2 ellipse-pair" id="mirPanel">
      <div class="panel">
        <h2>Before ($x$ and target $y$)</h2>
        <canvas id="mirIn" width="280" height="280" aria-label="Before reflection"></canvas>
        <p class="hint">Orange = $x$. Dashed blue = $y$.</p>
      </div>
      <div class="panel">
        <h2>After ($Hx$ vs $y$)</h2>
        <canvas id="mirOut" width="280" height="280" aria-label="After reflection"></canvas>
        <p class="hint">Solid orange = $Hx$. Dashed blue = $y$.</p>
      </div>
    </div>
    <p class="hunt-banner" id="mirBanner" hidden>$Hx = y$ — you found the $n$ for this reflection.</p>
    <p class="hint" id="mirHint"></p>
  </section>

  <section class="demo-block" aria-label="Aim onto axis">
    <h2>Why aim onto an axis?</h2>
    <p class="demo-intro">
      Suppose you have a vector $a$ pointing every which way, and you wish it lined up with the
      first coordinate axis — all other coordinates zero — <em>without</em> changing its length.
      Then later algebra only has to care about one number ($\\|a\\|$) instead of a whole list of
      coupled entries. A reflection can do exactly that: pick the target
      $t = (\\|a\\|, 0)$ and find $n$ so $Ha = t$. People call that choice a Householder reflection;
      the point is the goal, not the name.
    </p>
    <p class="demo-intro">
      Find such an $n$ with the slider. Afterward we write the formula that builds $n$ from $a$
      and $t$ automatically.
    </p>
    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Arrow $a$ angle <strong id="aimAngVal">55°</strong></span>
          <input id="aimAng" type="range" min="-170" max="170" step="1" value="55" />
        </label>
        <label class="slider">
          <span class="slider-label">Arrow $a$ length <strong id="aimLenVal">1.40</strong></span>
          <input id="aimLen" type="range" min="0.4" max="2.2" step="0.05" value="1.4" />
        </label>
      </div>
      <div class="control-actions">
        <div class="btns">
          <button id="aimChallenge" type="button">Another $a$</button>
        </div>
        <p class="help">New vector $a$; resets the mirror away from the solution.</p>
      </div>
    </div>

    <div class="panel hunt-panel" id="huntPanel">
      <h2>Find $n$ so $Ha$ has height $0$</h2>
      <div class="controls" style="margin-bottom:0.5rem">
        <div class="control-row">
          <label class="slider">
            <span class="slider-label">Mirror angle (sets $n$) <strong id="huntAngVal">20°</strong></span>
            <input id="huntAng" type="range" min="-90" max="90" step="1" value="20" />
          </label>
          <p class="help">Height of $Ha$ is $|(Ha)_2|$. Drive it to zero.</p>
        </div>
      </div>
      <div class="hunt-meter" aria-hidden="true">
        <span class="hunt-meter-label">$|(Ha)_2|$</span>
        <div class="hunt-meter-track">
          <div class="hunt-meter-fill" id="huntMeterFill"></div>
        </div>
        <strong class="hunt-meter-val" id="huntMeterVal">—</strong>
      </div>
      <canvas id="huntCanvas" width="360" height="360" aria-label="Find n so Ha is on axis"></canvas>
      <p class="hunt-banner" id="huntBanner" hidden>$Ha$ is on the axis — one number left, the rest are zeros.</p>
      <p class="hint" id="huntHint"></p>
    </div>

    <div class="panel" id="aimSolutionPanel">
      <h2>Construction of that $n$</h2>
      <p class="help" id="aimSolutionTease">
        Get $|(Ha)_2| = 0$ above; then we show the formula for $n$.
      </p>
      <div id="aimSolutionBody" hidden>
        <p class="demo-intro" style="margin-top:0">
          If the target is $t = (\\|a\\|, 0)$, the normal points along $a - t$
          (perpendicular bisector of the segment from $a$ to $t$). Normalize for unit $n$,
          then $H = I - 2nn^{\\top}$ sends $a$ to $t$.
        </p>
        <div class="math">
          $$n = \\frac{a - t}{\\|a - t\\|},\\qquad t = (\\|a\\|, 0)$$
        </div>
        <canvas id="aimCanvas" width="360" height="360" aria-label="Constructed Householder mirror"></canvas>
        <p class="hint" id="aimReadout"></p>
      </div>
    </div>
  </section>

  <section class="demo-block" aria-label="Larger matrix Householder">
    <h2>Same idea on a matrix column</h2>
    <p class="demo-intro">
      A matrix column is just a taller vector. If you want the first column of $A$ to look like
      $(\\|\\mathrm{col}_1\\|, 0, \\ldots, 0)^{\\top}$ — zeros under the pivot, length kept — apply one
      reflection $H$ built from that column and replace $A$ by $HA$. Everything else in $A$ moves
      too (same $H$ on every column), but you bought a simpler first column. The heatmap and stem
      plot show the zeros; the $n$D mirror is not drawn.
    </p>
    <div class="controls">
      <div class="control-actions">
        <div class="btns">
          <button id="hhRegen" type="button">New random $A$</button>
        </div>
        <p class="help">Gaussian $5\\times 5$. Build $H$ from column 1 and form $HA$.</p>
      </div>
    </div>
    <div class="grid-2">
      <div class="panel">
        <h2>$A$</h2>
        <canvas id="hhA" width="160" height="160" aria-label="Heatmap of A"></canvas>
      </div>
      <div class="panel">
        <h2>$HA$</h2>
        <canvas id="hhHA" width="160" height="160" aria-label="Heatmap of HA"></canvas>
      </div>
    </div>
    <div class="grid-2">
      <div class="panel">
        <h2>Column 1 before</h2>
        <canvas id="hhStemIn" width="220" height="160" aria-label="Stem plot column 1 before"></canvas>
      </div>
      <div class="panel">
        <h2>Column 1 after</h2>
        <canvas id="hhStemOut" width="220" height="160" aria-label="Stem plot column 1 after"></canvas>
      </div>
    </div>
  </section>

  <section class="demo-block" aria-label="Ellipse under left Householder">
    <h2>Circle → ellipse under a left reflection</h2>
    <p class="demo-intro">
      Under $A$, the unit circle becomes an ellipse (chapter 1).
      A left Householder $H$ reflects that ellipse; axis lengths stay put.
      Blend from before to after to see the flip. The image of $(1,0)$ lands on the $x$-axis.
    </p>
    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">$a_{11}$ <strong id="eA11Val">1.50</strong></span>
          <input id="eA11" type="range" min="-2.5" max="2.5" step="0.05" value="1.5" />
        </label>
        <label class="slider">
          <span class="slider-label">$a_{12}$ <strong id="eA12Val">1.00</strong></span>
          <input id="eA12" type="range" min="-2.5" max="2.5" step="0.05" value="1.0" />
        </label>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">$a_{21}$ <strong id="eA21Val">0.80</strong></span>
          <input id="eA21" type="range" min="-2.5" max="2.5" step="0.05" value="0.8" />
        </label>
        <label class="slider">
          <span class="slider-label">$a_{22}$ <strong id="eA22Val">1.20</strong></span>
          <input id="eA22" type="range" min="-2.5" max="2.5" step="0.05" value="1.2" />
        </label>
      </div>
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Blend $A \\to HA$ <strong id="ellBlendVal">0.00</strong></span>
          <input id="ellBlend" type="range" min="0" max="1" step="0.01" value="0" />
        </label>
        <p class="help">Animation aid — the true Householder is blend $=1$, not a “partial” reflection.</p>
      </div>
    </div>
    <p class="status" id="ellReadout" aria-live="polite"></p>
    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Unit circle</h2>
        <canvas id="ellIn" width="300" height="300" aria-label="Unit circle inputs"></canvas>
        <p class="hint">Inputs of length 1. Orange: $(1,0)$.</p>
      </div>
      <div class="panel">
        <h2>Output ellipse</h2>
        <canvas id="ellOut" width="300" height="300" aria-label="Blended ellipse after left Householder"></canvas>
        <p class="hint">Interpolated between $A$'s ellipse and $HA$'s. Stretches unchanged.</p>
      </div>
    </div>
  </section>

  <section class="demo-block" aria-label="Left then right steps">
    <h2>Simplify from both sides</h2>
    <p class="demo-intro">
      Left Householder aims column 1 onto the axis. Then a right rotation (Givens) zeros the
      remaining off-diagonal in $2\\times 2$. What remains on the diagonal are the stretch amounts.
    </p>
    <div class="controls">
      <div class="control-actions">
        <div class="btns">
          <button id="stepLeft" type="button">Apply left $H$</button>
          <button id="stepRight" type="button" class="secondary">Apply right $G$</button>
          <button id="stepReset" type="button" class="secondary">Reset</button>
        </div>
        <p class="help" id="stepHelp">Start from the demo $2\\times 2$. Apply left, then right.</p>
      </div>
    </div>
    <p class="formula" id="stepMatrix" aria-live="polite"></p>
    <div class="grid-2 ellipse-pair">
      <div class="panel">
        <h2>Inputs</h2>
        <canvas id="stepIn" width="280" height="280" aria-label="Unit circle for step demo"></canvas>
        <p class="hint">Unit circle. Right multiply reassigns which inputs go where.</p>
      </div>
      <div class="panel">
        <h2>Current map</h2>
        <canvas id="stepOut" width="280" height="280" aria-label="Ellipse after left/right steps"></canvas>
        <p class="hint">Left $H$ reflects the ellipse; right $G$ keeps the same output set.</p>
      </div>
    </div>
  </section>

  <section class="theory" aria-label="Assemble the product">
    <h2>Assemble the pieces</h2>
    <p>
      The orthogonal maps you applied on the left accumulate into an output reorientation $U$
      (rotations and reflections). Those on the right accumulate into an input reorientation $V$.
      The leftover diagonal entries (absolute values) are the stretch amounts $\\sigma_1, \\sigma_2$
      — the same axis lengths you saw on the ellipse.
    </p>
    <div class="math">
      $$A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$$
    </div>
    <p>
      That is chapter 1’s rotate → stretch → rotate again, written as a product.
      Reflections are allowed in the outer factors; lengths of the stretches are unchanged.
    </p>
  </section>

  <section class="demo-block" aria-label="Any size factors">
    <h2>Same pieces, any size</h2>
    <p class="demo-intro">
      For a random $n\\times n$ matrix the finished factors look the same: two orthogonal matrices
      and a list of nonnegative stretches. Libraries build them with Householder-style zeros first,
      then iteration on a simpler form — we show the finished product here.
    </p>
    <div class="controls">
      <div class="control-row">
        <label class="slider">
          <span class="slider-label">Size <em>n</em> <strong id="sizeVal">5</strong></span>
          <input id="size" type="range" min="0" max="10" step="1" value="2" />
        </label>
        <p class="help">Side length of random square $A$ (log-spaced stops $3\\ldots 32$).</p>
      </div>
      <div class="control-actions">
        <div class="btns">
          <button id="regen" type="button">Regenerate $A$</button>
        </div>
        <p class="help">New Gaussian matrix; recompute factors and product error.</p>
      </div>
    </div>
    <div class="shared">
      <div class="panel">
        <h2>Matrix $A$</h2>
        <canvas id="A" width="160" height="160"></canvas>
      </div>
    </div>
    <div class="panel" id="svdCol">
      <h2>Factors of $A$</h2>
      <div class="factor-row">
        <div class="factor"><span>Output $U$</span><canvas id="svdU" width="100" height="100"></canvas></div>
        <div class="factor"><span>Stretches</span><canvas id="svdS" width="100" height="80"></canvas></div>
        <div class="factor"><span>Input $V$</span><canvas id="svdV" width="100" height="100"></canvas></div>
        <div class="factor"><span>Product</span><canvas id="svdRecon" width="120" height="120"></canvas></div>
      </div>
      <p class="note" id="svdErr"></p>
    </div>
  </section>

  <section class="theory" aria-label="Name">
    <h2>What people call this</h2>
    <p>
      This factorization is the
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD). The stretch amounts $\\sigma_j$ are the <strong>singular values</strong>.
      Nothing new was added — only a name for the three-piece story you already checked with mirrors.
    </p>
  </section>

  <section class="conclusion" id="conclusion" aria-label="Summary">
    <h2>In short</h2>
    <p>
      Reflections preserve lengths like rotations. A Householder chooses a mirror so a column lands
      on an axis — that introduces zeros. Left and right orthogonal maps simplify $A$ until the
      diagonal holds the stretch amounts; the accumulated maps are the outer factors.
      Written $A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$, that package is the SVD.
    </p>
    <p class="next-chapter">
      Next you can keep only the largest stretches and drop the rest.
      <a href="./truncate.html">Continue →</a>
    </p>
  </section>
`;

/* ── DOM refs ─────────────────────────────────────────────────────────── */

const el = {
  mirAng: app.querySelector<HTMLInputElement>("#mirAng")!,
  mirAngVal: app.querySelector<HTMLElement>("#mirAngVal")!,
  mirIn: app.querySelector<HTMLCanvasElement>("#mirIn")!,
  mirOut: app.querySelector<HTMLCanvasElement>("#mirOut")!,
  mirMeterFill: app.querySelector<HTMLElement>("#mirMeterFill")!,
  mirMeterVal: app.querySelector<HTMLElement>("#mirMeterVal")!,
  mirBanner: app.querySelector<HTMLElement>("#mirBanner")!,
  mirHint: app.querySelector<HTMLElement>("#mirHint")!,
  mirPanel: app.querySelector<HTMLElement>("#mirPanel")!,
  buildAng: app.querySelector<HTMLInputElement>("#buildAng")!,
  buildAngVal: app.querySelector<HTMLElement>("#buildAngVal")!,
  build1: app.querySelector<HTMLCanvasElement>("#build1")!,
  build2: app.querySelector<HTMLCanvasElement>("#build2")!,
  build3: app.querySelector<HTMLCanvasElement>("#build3")!,
  buildFormula: app.querySelector<HTMLElement>("#buildFormula")!,

  aimAng: app.querySelector<HTMLInputElement>("#aimAng")!,
  aimAngVal: app.querySelector<HTMLElement>("#aimAngVal")!,
  aimLen: app.querySelector<HTMLInputElement>("#aimLen")!,
  aimLenVal: app.querySelector<HTMLElement>("#aimLenVal")!,
  aimChallenge: app.querySelector<HTMLButtonElement>("#aimChallenge")!,
  aimCanvas: app.querySelector<HTMLCanvasElement>("#aimCanvas")!,
  aimReadout: app.querySelector<HTMLElement>("#aimReadout")!,
  aimSolutionTease: app.querySelector<HTMLElement>("#aimSolutionTease")!,
  aimSolutionBody: app.querySelector<HTMLElement>("#aimSolutionBody")!,
  huntPanel: app.querySelector<HTMLElement>("#huntPanel")!,
  huntAng: app.querySelector<HTMLInputElement>("#huntAng")!,
  huntAngVal: app.querySelector<HTMLElement>("#huntAngVal")!,
  huntCanvas: app.querySelector<HTMLCanvasElement>("#huntCanvas")!,
  huntHint: app.querySelector<HTMLElement>("#huntHint")!,
  huntBanner: app.querySelector<HTMLElement>("#huntBanner")!,
  huntMeterFill: app.querySelector<HTMLElement>("#huntMeterFill")!,
  huntMeterVal: app.querySelector<HTMLElement>("#huntMeterVal")!,

  hhRegen: app.querySelector<HTMLButtonElement>("#hhRegen")!,
  hhA: app.querySelector<HTMLCanvasElement>("#hhA")!,
  hhHA: app.querySelector<HTMLCanvasElement>("#hhHA")!,
  hhStemIn: app.querySelector<HTMLCanvasElement>("#hhStemIn")!,
  hhStemOut: app.querySelector<HTMLCanvasElement>("#hhStemOut")!,

  eA11: app.querySelector<HTMLInputElement>("#eA11")!,
  eA12: app.querySelector<HTMLInputElement>("#eA12")!,
  eA21: app.querySelector<HTMLInputElement>("#eA21")!,
  eA22: app.querySelector<HTMLInputElement>("#eA22")!,
  eA11Val: app.querySelector<HTMLElement>("#eA11Val")!,
  eA12Val: app.querySelector<HTMLElement>("#eA12Val")!,
  eA21Val: app.querySelector<HTMLElement>("#eA21Val")!,
  eA22Val: app.querySelector<HTMLElement>("#eA22Val")!,
  ellBlend: app.querySelector<HTMLInputElement>("#ellBlend")!,
  ellBlendVal: app.querySelector<HTMLElement>("#ellBlendVal")!,
  ellIn: app.querySelector<HTMLCanvasElement>("#ellIn")!,
  ellOut: app.querySelector<HTMLCanvasElement>("#ellOut")!,
  ellReadout: app.querySelector<HTMLElement>("#ellReadout")!,

  stepLeft: app.querySelector<HTMLButtonElement>("#stepLeft")!,
  stepRight: app.querySelector<HTMLButtonElement>("#stepRight")!,
  stepReset: app.querySelector<HTMLButtonElement>("#stepReset")!,
  stepHelp: app.querySelector<HTMLElement>("#stepHelp")!,
  stepMatrix: app.querySelector<HTMLElement>("#stepMatrix")!,
  stepIn: app.querySelector<HTMLCanvasElement>("#stepIn")!,
  stepOut: app.querySelector<HTMLCanvasElement>("#stepOut")!,

  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  regen: app.querySelector<HTMLButtonElement>("#regen")!,
  A: app.querySelector<HTMLCanvasElement>("#A")!,
  svdU: app.querySelector<HTMLCanvasElement>("#svdU")!,
  svdS: app.querySelector<HTMLCanvasElement>("#svdS")!,
  svdV: app.querySelector<HTMLCanvasElement>("#svdV")!,
  svdRecon: app.querySelector<HTMLCanvasElement>("#svdRecon")!,
  svdErr: app.querySelector<HTMLParagraphElement>("#svdErr")!,
};

/* ── Canvas helpers ────────────────────────────────────────────────────── */

function toCanvas(
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
): [number, number] {
  return [cx + x * scale, cy - y * scale];
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
  color: string,
  label: string,
): void {
  const [x0, y0] = toCanvas(cx, cy, scale, 0, 0);
  const [x1, y1] = toCanvas(cx, cy, scale, x, y);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  const ux = dx / len;
  const uy = dy / len;
  const head = 9;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.25;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * ux + 0.5 * head * uy, y1 - head * uy - 0.5 * head * ux);
  ctx.lineTo(x1 - head * ux - 0.5 * head * uy, y1 - head * uy + 0.5 * head * ux);
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, x1 + 7 * ux, y1 + 7 * uy);
  }
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  sample: (t: number) => [number, number],
  color: string,
  width = 2,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const t = (i / 180) * Math.PI * 2;
    const [x, y] = sample(t);
    const [px, py] = toCanvas(cx, cy, scale, x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawAxes(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  worldR: number,
): void {
  const r = worldR * scale;
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();
  ctx.fillStyle = MUTED;
  ctx.font = "500 11px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("x", cx + r - 12, cy + 14);
  ctx.textAlign = "center";
  ctx.fillText("y", cx + 10, cy - r + 12);
}

function drawMirrorLine(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  lineAngleDeg: number,
  worldR: number,
  opts?: { label?: boolean },
): void {
  const th = (lineAngleDeg * Math.PI) / 180;
  const dx = Math.cos(th);
  const dy = Math.sin(th);
  const r = worldR * 1.15;
  const [x0, y0] = toCanvas(cx, cy, scale, -r * dx, -r * dy);
  const [x1, y1] = toCanvas(cx, cy, scale, r * dx, r * dy);

  // Soft “glass” band so the mirror reads as a surface, not another axis
  const nx = -dy;
  const ny = dx;
  const band = 5;
  ctx.fillStyle = "rgba(213, 94, 0, 0.12)";
  ctx.beginPath();
  ctx.moveTo(x0 + nx * band, y0 - ny * band);
  ctx.lineTo(x1 + nx * band, y1 - ny * band);
  ctx.lineTo(x1 - nx * band, y1 + ny * band);
  ctx.lineTo(x0 - nx * band, y0 + ny * band);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = MIRROR;
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  if (opts?.label !== false) {
    const [lx, ly] = toCanvas(cx, cy, scale, 0.85 * r * dx, 0.85 * r * dy);
    ctx.fillStyle = MIRROR;
    ctx.font = "700 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Offset label off the line along the normal
    ctx.fillText("mirror", lx + nx * 14, ly - ny * 14);
    ctx.textBaseline = "alphabetic";
  }
}

function paintCanvas(
  canvas: HTMLCanvasElement,
  worldR: number,
  drawContent: (ctx: CanvasRenderingContext2D, cx: number, cy: number, scale: number) => void,
): void {
  const { cssW, ctx } = prepareHiDpiCanvas(canvas);
  if (!ctx) return;
  const w = cssW;
  ctx.clearRect(0, 0, w, w);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, w);
  const cx = w / 2;
  const cy = w / 2;
  const scale = (0.4 * w) / worldR;
  drawAxes(ctx, cx, cy, scale, worldR);
  drawContent(ctx, cx, cy, scale);
}

function drawStem(
  canvas: HTMLCanvasElement,
  values: number[],
  color: string,
): void {
  const { cssW, cssH, ctx } = prepareHiDpiCanvas(canvas);
  if (!ctx) return;
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 28;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const maxAbsV = Math.max(...values.map((v) => Math.abs(v)), 1e-6);
  const midY = padT + plotH / 2;
  const n = values.length;
  const gap = plotW / Math.max(n, 1);

  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, midY);
  ctx.lineTo(padL + plotW, midY);
  ctx.stroke();

  ctx.font = "600 11px DM Sans, system-ui, sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const x = padL + gap * (i + 0.5);
    const h = (values[i] / maxAbsV) * (plotH * 0.42);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, midY);
    ctx.lineTo(x, midY - h);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, midY - h, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = MUTED;
    ctx.fillText(String(i + 1), x, cssH - 8);
  }
}

function fmt(x: number, d = 2): string {
  return x.toFixed(d);
}

/** Fixed-width signed number so neighboring labels don't jump when the sign flips. */
function fmtSigned(x: number, d = 3): string {
  const body = Math.abs(x).toFixed(d);
  return (x < 0 ? "-" : "\u2007") + body;
}

function fmtPair(x: number, y: number, d = 3): string {
  return `(${fmtSigned(x, d)}, ${fmtSigned(y, d)})`;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/* ── Build reflection operator ─────────────────────────────────────────── */

const BUILD_PROBE: [number, number] = [1.1, 0.55];
const NORMAL_COLOR = "#009E73";

const DECOMP_COLOR = "#7B2D8E";

function paintBuild(): void {
  const ang = Number(el.buildAng.value);
  el.buildAngVal.textContent = `${ang}°`;
  const [nx, ny] = normalFromLineAngle(ang);
  const H = householderFromNormal(nx, ny);
  const [x0, y0] = BUILD_PROBE;
  const [hx, hy] = applyMat2(H, x0, y0);
  const { dot, parallel, normal } = decomposeAlongNormal(x0, y0, nx, ny);
  const [xp, yp] = parallel;
  const [xn, yn] = normal;

  el.buildFormula.textContent =
    `n ≈ ${fmtPair(nx, ny)} · ` +
    `nᵀx ≈ ${fmtSigned(dot)} · ` +
    `xₙ ≈ ${fmtPair(xn, yn)} · ` +
    `Hx ≈ ${fmtPair(hx, hy)}`;

  const extent = 1.6;

  paintCanvas(el.build1, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, nx, ny, NORMAL_COLOR, "n");
  });

  paintCanvas(el.build2, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, xp, yp, DECOMP_COLOR, "x∥");
    // normal piece from tip of parallel to tip of x
    const [p0x, p0y] = toCanvas(cx, cy, scale, xp, yp);
    const [p1x, p1y] = toCanvas(cx, cy, scale, x0, y0);
    ctx.strokeStyle = NORMAL_COLOR;
    ctx.fillStyle = NORMAL_COLOR;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(p0x, p0y);
    ctx.lineTo(p1x, p1y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p1x, p1y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.fillText("xₙ", p1x + 6, p1y - 6);
    drawArrow(ctx, cx, cy, scale, x0, y0, ACCENT, "x");
    drawArrow(ctx, cx, cy, scale, nx * 0.55, ny * 0.55, NORMAL_COLOR, "n");
  });

  paintCanvas(el.build3, extent, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, extent);
    drawArrow(ctx, cx, cy, scale, x0, y0, ACCENT, "x");
    drawArrow(ctx, cx, cy, scale, hx, hy, ACCENT2, "Hx");
    ctx.strokeStyle = MUTED;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const [a0, a1] = toCanvas(cx, cy, scale, x0, y0);
    const [b0, b1] = toCanvas(cx, cy, scale, hx, hy);
    ctx.moveTo(a0, a1);
    ctx.lineTo(b0, b1);
    ctx.stroke();
    ctx.setLineDash([]);
    // midpoint on mirror (foot)
    const [fx, fy] = toCanvas(cx, cy, scale, xp, yp);
    ctx.fillStyle = MUTED;
    ctx.beginPath();
    ctx.arc(fx, fy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

/* ── Reflect across a line (match target) ──────────────────────────────── */

/** Orange arrow to flip; dashed blue = where it should land. */
const MIR_SRC: [number, number] = [1, 0];
const MIR_TGT: [number, number] = [0, 1];

function drawDashedArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number,
  color: string,
  label: string,
): void {
  const [x0, y0] = toCanvas(cx, cy, scale, 0, 0);
  const [x1, y1] = toCanvas(cx, cy, scale, x, y);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  const ux = dx / len;
  const uy = dy / len;
  const head = 9;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.25;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * ux + 0.5 * head * uy, y1 - head * uy - 0.5 * head * ux);
  ctx.lineTo(x1 - head * ux - 0.5 * head * uy, y1 - head * uy + 0.5 * head * ux);
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "600 12px DM Sans, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, x1 + 7 * ux, y1 + 7 * uy);
  }
}

function paintMirror(): void {
  const ang = Number(el.mirAng.value);
  el.mirAngVal.textContent = `${ang}°`;
  const [nx, ny] = normalFromLineAngle(ang);
  const H = householderFromNormal(nx, ny);
  const [rx, ry] = applyMat2(H, MIR_SRC[0], MIR_SRC[1]);
  const dist = Math.hypot(rx - MIR_TGT[0], ry - MIR_TGT[1]);
  const won = dist < 0.08;

  const meter = Math.min(1, dist / 1.5);
  el.mirMeterFill.style.transform = `scaleX(${meter})`;
  el.mirMeterFill.classList.toggle("near", meter < 0.25 && !won);
  el.mirMeterFill.classList.toggle("won", won);
  el.mirMeterVal.textContent = fmt(dist, 3);
  el.mirBanner.hidden = !won;
  el.mirPanel.classList.toggle("hunt-won", won);
  el.mirHint.textContent = won
    ? "Hx = y with the n from this mirror. Lengths: ‖Hx‖ = ‖x‖."
    : "Adjust n until ‖Hx − y‖ is near zero (right panel).";

  paintCanvas(el.mirIn, 1.4, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, ang, 1.4);
    drawCurve(ctx, cx, cy, scale, (t) => [Math.cos(t), Math.sin(t)], INK);
    drawArrow(ctx, cx, cy, scale, MIR_SRC[0], MIR_SRC[1], ACCENT2, "start");
    drawDashedArrow(ctx, cx, cy, scale, MIR_TGT[0], MIR_TGT[1], ACCENT, "target");
  });

  paintCanvas(el.mirOut, 1.4, (ctx, cx, cy, scale) => {
    if (won) {
      ctx.strokeStyle = "#009E73";
      ctx.lineWidth = 3;
      ctx.beginPath();
      const [ax0, ay0] = toCanvas(cx, cy, scale, 0, 0);
      const [ax1, ay1] = toCanvas(cx, cy, scale, MIR_TGT[0], MIR_TGT[1]);
      ctx.moveTo(ax0, ay0);
      ctx.lineTo(ax1, ay1);
      ctx.stroke();
    }
    drawMirrorLine(ctx, cx, cy, scale, ang, 1.4);
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (t) => reflectAcrossNormal(Math.cos(t), Math.sin(t), nx, ny),
      INK,
    );
    drawDashedArrow(ctx, cx, cy, scale, MIR_TGT[0], MIR_TGT[1], ACCENT, "target");
    drawArrow(ctx, cx, cy, scale, rx, ry, won ? "#009E73" : ACCENT2, "reflected");
  });
}

/* ── Aim onto axis (challenge) ─────────────────────────────────────────── */

let huntWon = false;

function aimVector(): [number, number] {
  const th = (Number(el.aimAng.value) * Math.PI) / 180;
  const len = Number(el.aimLen.value);
  return [len * Math.cos(th), len * Math.sin(th)];
}

/** Offset hunt mirror from the true Householder so the challenge is not already solved. */
function resetHuntAwayFromAnswer(answerDeg: number): void {
  let offset = 35 + Math.floor(Math.random() * 40);
  if (Math.random() < 0.5) offset = -offset;
  let h = answerDeg + offset;
  while (h > 90) h -= 180;
  while (h < -90) h += 180;
  el.huntAng.value = String(Math.round(h));
  huntWon = false;
}

function newAimChallenge(): void {
  el.aimAng.value = String(Math.round(-150 + Math.random() * 300));
  el.aimLen.value = (0.6 + Math.random() * 1.4).toFixed(2);
  const [ax, ay] = aimVector();
  const aimed = householderAimToE1(ax, ay);
  resetHuntAwayFromAnswer(aimed.mirrorAngleDeg);
  paintAim();
}

function paintAim(): void {
  const ang = Number(el.aimAng.value);
  const len = Number(el.aimLen.value);
  el.aimAngVal.textContent = `${ang}°`;
  el.aimLenVal.textContent = fmt(len);

  const [ax, ay] = aimVector();
  const aimed = householderAimToE1(ax, ay);
  const worldR = Math.max(2.0, aimed.normA * 1.25);

  const hunt = Number(el.huntAng.value);
  el.huntAngVal.textContent = `${hunt}°`;
  const [hnx, hny] = normalFromLineAngle(hunt);
  const [rx, ry] = reflectAcrossNormal(ax, ay, hnx, hny);
  const height = Math.abs(ry);
  const thresh = 0.045 * Math.max(aimed.normA, 1);
  const onAxis = height < thresh;
  if (onAxis && !huntWon) huntWon = true;

  // Proximity meter: full when far, empty when on axis
  const meter = Math.min(1, height / Math.max(aimed.normA, 0.5));
  el.huntMeterFill.style.transform = `scaleX(${meter})`;
  el.huntMeterFill.classList.toggle("near", meter < 0.25 && !onAxis);
  el.huntMeterFill.classList.toggle("won", onAxis);
  el.huntMeterVal.textContent = fmt(height, 3);
  el.huntPanel.classList.toggle("hunt-won", onAxis);
  el.huntBanner.hidden = !onAxis;

  paintCanvas(el.huntCanvas, worldR, (ctx, cx, cy, scale) => {
    if (onAxis) {
      // Glow the x-axis
      ctx.strokeStyle = "#009E73";
      ctx.lineWidth = 4;
      ctx.beginPath();
      const [x0, y0] = toCanvas(cx, cy, scale, -worldR, 0);
      const [x1, y1] = toCanvas(cx, cy, scale, worldR, 0);
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    drawMirrorLine(ctx, cx, cy, scale, hunt, worldR);
    drawArrow(ctx, cx, cy, scale, ax, ay, ACCENT2, "a");
    drawArrow(ctx, cx, cy, scale, rx, ry, onAxis ? "#009E73" : ACCENT, "reflected");
  });

  if (onAxis) {
    el.huntHint.textContent =
      `Ha ≈ (${fmt(rx, 3)}, ${fmt(ry, 3)}). This n matches n ∥ (a − t) below.`;
    el.aimSolutionTease.hidden = true;
    el.aimSolutionBody.hidden = false;
  } else {
    const warmth =
      meter < 0.15 ? "Almost: |(Ha)₂| is nearly 0." :
      meter < 0.35 ? "n is getting closer." :
      "Vary n; watch |(Ha)₂|.";
    el.huntHint.textContent = warmth;
    // Keep solution visible once unlocked for this arrow; re-lock only on new challenge
    if (!huntWon) {
      el.aimSolutionTease.hidden = false;
      el.aimSolutionBody.hidden = true;
    }
  }

  // Solution panel (always paint so canvas is ready when unlocked)
  paintCanvas(el.aimCanvas, worldR, (ctx, cx, cy, scale) => {
    drawMirrorLine(ctx, cx, cy, scale, aimed.mirrorAngleDeg, worldR);
    const [sx, sy] = toCanvas(cx, cy, scale, ax, ay);
    const [tx, ty] = toCanvas(cx, cy, scale, aimed.targetX, aimed.targetY);
    ctx.strokeStyle = MUTED;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    drawArrow(ctx, cx, cy, scale, ax, ay, ACCENT2, "a");
    drawArrow(ctx, cx, cy, scale, aimed.targetX, aimed.targetY, ACCENT, "target");
    const [hx, hy] = applyMat2(aimed.H, ax, ay);
    drawArrow(ctx, cx, cy, scale, hx, hy, INK, "Ha");
  });

  el.aimReadout.textContent =
    `t = (${fmt(aimed.targetX, 3)}, 0). n ≈ (${fmt(aimed.nx, 3)}, ${fmt(aimed.ny, 3)}). ` +
    `Ha ≈ (${fmt(aimed.targetX, 3)}, 0).`;
}

/* ── 5×5 Householder ───────────────────────────────────────────────────── */

let hhMat: Matrix;
let hhHA: Matrix;

function regenHH(): void {
  hhMat = randomNormal(5, 5, 1);
  const H = householderAimColumn(columnOf(hhMat, 0));
  hhHA = applyLeft(H, hhMat);
  paintHH();
}

function paintHH(): void {
  const scale = Math.max(maxAbs(hhMat), maxAbs(hhHA), 1e-6);
  drawHeatmap(el.hhA, hhMat, scale);
  drawHeatmap(el.hhHA, hhHA, scale);
  drawStem(el.hhStemIn, columnOf(hhMat, 0), ACCENT2);
  drawStem(el.hhStemOut, columnOf(hhHA, 0), ACCENT);
}

/* ── Ellipse under left HH ─────────────────────────────────────────────── */

function readEllipseA(): Matrix {
  return fromNested([
    [Number(el.eA11.value), Number(el.eA12.value)],
    [Number(el.eA21.value), Number(el.eA22.value)],
  ]);
}

function syncEllipseLabels(): void {
  el.eA11Val.textContent = fmt(Number(el.eA11.value));
  el.eA12Val.textContent = fmt(Number(el.eA12.value));
  el.eA21Val.textContent = fmt(Number(el.eA21.value));
  el.eA22Val.textContent = fmt(Number(el.eA22.value));
  el.ellBlendVal.textContent = fmt(Number(el.ellBlend.value));
}

function paintEllipse(): void {
  syncEllipseLabels();
  const A = readEllipseA();
  const col = columnOf(A, 0);
  const { H } = householderAimToE1(col[0], col[1]);
  const HA = applyLeft(H, A);
  const t = Number(el.ellBlend.value);
  const fA = frameFromMatrix(A);
  const fHA = frameFromMatrix(HA);
  const [s1, s2] = fA.sigma;
  const outR = Math.max(fA.outScale, fHA.outScale);

  paintCanvas(el.ellIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (u) => [Math.cos(u), Math.sin(u)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1,0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0,1)");
  });

  paintCanvas(el.ellOut, outR, (ctx, cx, cy, scale) => {
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (u) => {
        const c = Math.cos(u);
        const s = Math.sin(u);
        const aPt = applyMat2(A, c, s);
        const hPt = applyMat2(HA, c, s);
        return [
          aPt[0] * (1 - t) + hPt[0] * t,
          aPt[1] * (1 - t) + hPt[1] * t,
        ];
      },
      INK,
    );
    const a10 = applyMat2(A, 1, 0);
    const h10 = applyMat2(HA, 1, 0);
    const ix = a10[0] * (1 - t) + h10[0] * t;
    const iy = a10[1] * (1 - t) + h10[1] * t;
    drawArrow(ctx, cx, cy, scale, ix, iy, ACCENT2, "A(1,0)→");
    if (t > 0.85) {
      drawArrow(ctx, cx, cy, scale, h10[0], h10[1], ACCENT, "HA(1,0)");
    }
  });

  el.ellReadout.textContent =
    `Stretch amounts σ ≈ ${fmt(s1, 3)}, ${fmt(s2, 3)} — unchanged by left H ` +
    `(HA has σ ≈ ${fmt(fHA.sigma[0], 3)}, ${fmt(fHA.sigma[1], 3)}). ` +
    `At blend 1, image of (1,0) lies on the x-axis.`;
}

/* ── Left / right steps ────────────────────────────────────────────────── */

let stepM: Matrix;
let stepPhase: "start" | "left" | "done" = "start";

function fmtMat2(M: Matrix): string {
  return (
    `$$\\begin{pmatrix}` +
    `${fmt(get(M, 0, 0), 3)} & ${fmt(get(M, 0, 1), 3)} \\\\ ` +
    `${fmt(get(M, 1, 0), 3)} & ${fmt(get(M, 1, 1), 3)}` +
    `\\end{pmatrix}$$`
  );
}

function resetSteps(): void {
  stepM = copy(demoMatrix2());
  stepPhase = "start";
  el.stepHelp.textContent = "Start from the demo 2×2. Apply left H, then right G.";
  el.stepLeft.disabled = false;
  el.stepRight.disabled = true;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

function paintSteps(): void {
  const f = frameFromMatrix(stepM);
  const outR = f.outScale;

  paintCanvas(el.stepIn, 1.35, (ctx, cx, cy, scale) => {
    drawCurve(ctx, cx, cy, scale, (u) => [Math.cos(u), Math.sin(u)], INK);
    drawArrow(ctx, cx, cy, scale, 1, 0, ACCENT2, "(1,0)");
    drawArrow(ctx, cx, cy, scale, 0, 1, ACCENT, "(0,1)");
  });

  paintCanvas(el.stepOut, outR, (ctx, cx, cy, scale) => {
    drawCurve(
      ctx,
      cx,
      cy,
      scale,
      (u) => applyMat2(stepM, Math.cos(u), Math.sin(u)),
      INK,
    );
    const [s1, s2] = f.sigma;
    drawArrow(
      ctx,
      cx,
      cy,
      scale,
      get(f.U, 0, 0) * s1,
      get(f.U, 1, 0) * s1,
      ACCENT2,
      "σ₁",
    );
    if (s2 > 1e-6) {
      drawArrow(
        ctx,
        cx,
        cy,
        scale,
        get(f.U, 0, 1) * s2,
        get(f.U, 1, 1) * s2,
        ACCENT,
        "σ₂",
      );
    }
  });

  const phaseLabel =
    stepPhase === "start"
      ? "Current $A$"
      : stepPhase === "left"
        ? "After left $H$"
        : "After left $H$ and right $G$";
  el.stepMatrix.innerHTML = `${phaseLabel}: ${fmtMat2(stepM)}`;
}

function doStepLeft(): void {
  if (stepPhase !== "start") return;
  const { H } = householderAimToE1(get(stepM, 0, 0), get(stepM, 1, 0));
  stepM = applyLeft(H, stepM);
  stepPhase = "left";
  el.stepHelp.textContent = "Column 1 on the axis. Next: right Givens to zero the (1,2) entry.";
  el.stepLeft.disabled = true;
  el.stepRight.disabled = false;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

function doStepRight(): void {
  if (stepPhase !== "left") return;
  const G = rightGivensZeroSuperdiag(stepM);
  stepM = applyRight(stepM, G);
  stepPhase = "done";
  el.stepHelp.textContent =
    "Nearly diagonal — absolute diagonal entries match the ellipse stretches.";
  el.stepRight.disabled = true;
  paintSteps();
  void window.MathJax?.typesetPromise?.([el.stepMatrix]);
}

/* ── Any-size SVD heatmaps ─────────────────────────────────────────────── */

const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

let A: Matrix;
let svd: SvdResult;
let svdReconMat: Matrix;
let sharedScale = 1;
let sigmaScale = 1;

function sizeFromSlider(): number {
  const i = clamp(Math.round(Number(el.size.value) || 0), 0, SIZE_STOPS.length - 1);
  el.size.value = String(i);
  return SIZE_STOPS[i];
}

function syncSliderLabels(): void {
  el.sizeVal.textContent = String(sizeFromSlider());
}

function recompute(newA: boolean): void {
  const n = sizeFromSlider();
  syncSliderLabels();
  if (newA || !A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, n);
  svdReconMat = reconstruct(svd.U, svd.sigma, svd.V);
  sharedScale = Math.max(maxAbs(A), maxAbs(svdReconMat), 1e-6);
  sigmaScale = Math.max(...svd.sigma, 1e-6);
  paintSvd();
}

function paintSvd(): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdReconMat, sharedScale);
  const err = frobeniusSq(sub(A, svdReconMat));
  el.svdErr.textContent = `‖A − product‖_F² = ${err.toExponential(3)}`;
}

/* ── Master paint + listeners ──────────────────────────────────────────── */

function paintAll(): void {
  paintBuild();
  paintMirror();
  paintAim();
  paintHH();
  paintEllipse();
  paintSteps();
  paintSvd();
}

el.buildAng.addEventListener("input", paintBuild);
el.mirAng.addEventListener("input", paintMirror);

el.aimChallenge.addEventListener("click", newAimChallenge);
el.aimAng.addEventListener("input", () => {
  huntWon = false;
  paintAim();
});
el.aimLen.addEventListener("input", () => {
  huntWon = false;
  paintAim();
});
el.huntAng.addEventListener("input", paintAim);

el.hhRegen.addEventListener("click", regenHH);

for (const input of [el.eA11, el.eA12, el.eA21, el.eA22, el.ellBlend]) {
  input.addEventListener("input", paintEllipse);
}

el.stepLeft.addEventListener("click", doStepLeft);
el.stepRight.addEventListener("click", doStepRight);
el.stepReset.addEventListener("click", resetSteps);

el.regen.addEventListener("click", () => recompute(true));
el.size.addEventListener("input", () => recompute(true));

regenHH();
resetSteps();
syncSliderLabels();
recompute(true);
{
  const [ax, ay] = aimVector();
  resetHuntAwayFromAnswer(householderAimToE1(ax, ay).mirrorAngleDeg);
}
paintAll();

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});
