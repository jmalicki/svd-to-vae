import "./style.css";
import { classicalSvd, type SvdResult } from "./classicalSvd";
import {
  SvdGradTrainer,
  probeGpu,
  type DeviceKind,
  type GradState,
} from "./svdGrad";
import {
  type Matrix,
  randomNormal,
  reconstruct,
  maxAbs,
  frobeniusSq,
  sub,
} from "./matrix";
import { drawHeatmap, drawSigmaBars, drawLossChart, type LossPoint } from "./viz";

declare global {
  interface Window {
    MathJax?: { typesetPromise?: (els?: Element[]) => Promise<void> };
  }
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header>
    <h1>SVD via gradient descent</h1>
    <p class="repo">
      <a href="https://github.com/jmalicki/svd-grad" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
    </p>
    <p class="lede">
      The
      <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition" target="_blank" rel="noopener noreferrer">singular value decomposition</a>
      (SVD) factors a matrix $A$ as $U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$ with
      <a href="https://en.wikipedia.org/wiki/Orthonormality" target="_blank" rel="noopener noreferrer">orthonormal</a>
      $U$, $V$ (columns of length $1$, and mutually perpendicular) and nonnegative singular
      values $\\sigma$ — the standard tool for low-rank approximation, PCA, and many other
      matrix problems.
      This page shows that you can recover a truncated SVD by plain gradient descent on
      reconstruction error, snapping $U$ and $V$ back to orthonormal columns after each step.
    </p>
  </header>

  <section class="theory" aria-label="Loss construction">
    <h2>How it works</h2>
    <ol class="theory-steps">
      <li>
        <p>
          By the
          <a href="https://en.wikipedia.org/wiki/Low-rank_approximation#Eckart%E2%80%93Young%E2%80%93Mirsky_theorem" target="_blank" rel="noopener noreferrer">Eckart–Young–Mirsky theorem</a>,
          the best rank-$k$ approximation to $A$ in Frobenius norm is the truncated SVD
          $\\hat A_k = U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}$.
        </p>
        <div class="math">
          $$\\min_{\\mathrm{rank}(B)\\le k}\\;\\|A - B\\|_F
            = \\|A - U_k\\,\\mathrm{diag}(\\sigma)\\,V_k^{\\top}\\|_F$$
        </div>
      </li>
      <li>
        <p>
          So write a reconstruction objective over factors
          $U\\in\\mathbb{R}^{n\\times k}$, $\\sigma\\in\\mathbb{R}^{k}_{\\ge 0}$, $V\\in\\mathbb{R}^{n\\times k}$:
        </p>
        <div class="math">
          $$L_{\\mathrm{recon}} = \\bigl\\|A - U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}\\bigr\\|_F^{2}$$
        </div>
        <p>
          Singular values are nonnegative, so we do not treat $\\sigma$ itself as a free
          Euclidean parameter. Instead we store an unconstrained vector
          $\\mathrm{raw}\\in\\mathbb{R}^{k}$ and set
        </p>
        <div class="math">
          $$\\sigma_j = \\mathrm{softplus}(\\mathrm{raw}_j)
            = \\ln\\bigl(1+e^{\\mathrm{raw}_j}\\bigr) \\gt 0.$$
        </div>
        <p class="theory-note">
          <a href="https://en.wikipedia.org/wiki/Softplus" target="_blank" rel="noopener noreferrer">Softplus</a>
          is a smooth map $\\mathbb{R}\\to(0,\\infty)$: large negative inputs become tiny positives,
          large positive inputs stay roughly unchanged. Gradients flow through it into
          $\\mathrm{raw}$. The trainable factors are then $U$, $\\mathrm{raw}$, and $V$;
          displayed $\\sigma$ bars are $\\mathrm{softplus}(\\mathrm{raw})$.
          (Why not $|\\mathrm{raw}|$? See
          <a href="#appendix-softplus">Why softplus</a>.)
        </p>
        <p class="theory-note">
          Orthogonality is <strong>not</strong> required for a best low-rank fit — many factorizations
          can realize the same $\\hat A_k$. Minimizing $L_{\\mathrm{recon}}$ alone targets that
          reconstruction, but not necessarily the SVD bases $U$ and $V$.
          By Eckart–Young, the best attainable value is the constant
          $\\|A - \\hat A_k\\|_F^{2}$ from truncated SVD (the dashed floor on the loss chart).
        </p>
        <figure class="nn-figure" aria-label="Rank-k factorization as a linear network">
          <svg viewBox="0 0 640 250" role="img" xmlns="http://www.w3.org/2000/svg">
            <title>Gradient-descent SVD as a three-layer linear network</title>
            <desc>
              Input vector x in R^n goes through V transpose into a k-dimensional bottleneck,
              is scaled by singular values sigma, then through U to produce A-hat x.
              The loss compares A-hat x to A x.
            </desc>
            <defs>
              <marker id="nnArrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 1.5 L 9 5 L 0 8.5 Z" fill="#6b6b6b"/>
              </marker>
            </defs>

            <rect x="24" y="52" width="88" height="168" rx="10" fill="#f4f7fa" stroke="#c5d4e0"/>
            <rect x="236" y="74" width="88" height="124" rx="10" fill="#fbf6eb" stroke="#e0d0a8"/>
            <rect x="448" y="52" width="88" height="168" rx="10" fill="#f4f7fa" stroke="#c5d4e0"/>

            <text x="68" y="38" text-anchor="middle" fill="#6b6b6b"
              font-family="DM Sans, system-ui, sans-serif" font-size="12" font-weight="600">input x</text>
            <text x="280" y="38" text-anchor="middle" fill="#6b6b6b"
              font-family="DM Sans, system-ui, sans-serif" font-size="12" font-weight="600">bottleneck</text>
            <text x="492" y="38" text-anchor="middle" fill="#6b6b6b"
              font-family="DM Sans, system-ui, sans-serif" font-size="12" font-weight="600">output Âx</text>

            <circle cx="68" cy="80" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <circle cx="68" cy="114" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <circle cx="68" cy="148" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <text x="68" y="174" text-anchor="middle" fill="#6b6b6b"
              font-family="IBM Plex Mono, monospace" font-size="14">⋮</text>
            <circle cx="68" cy="196" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <text x="68" y="238" text-anchor="middle" fill="#2d2d2d"
              font-family="DM Sans, system-ui, sans-serif" font-size="13">n units</text>

            <circle cx="280" cy="104" r="14" fill="#fff" stroke="#e69f00" stroke-width="2.2"/>
            <circle cx="280" cy="148" r="14" fill="#fff" stroke="#e69f00" stroke-width="2.2"/>
            <circle cx="280" cy="192" r="14" fill="#fff" stroke="#e69f00" stroke-width="2.2"/>
            <text x="280" y="108.5" text-anchor="middle" fill="#2d2d2d"
              font-family="IBM Plex Mono, monospace" font-size="11" font-weight="500">1</text>
            <text x="280" y="152.5" text-anchor="middle" fill="#2d2d2d"
              font-family="IBM Plex Mono, monospace" font-size="11" font-weight="500">2</text>
            <text x="280" y="196.5" text-anchor="middle" fill="#2d2d2d"
              font-family="IBM Plex Mono, monospace" font-size="11" font-weight="500">3</text>
            <text x="280" y="238" text-anchor="middle" fill="#2d2d2d"
              font-family="DM Sans, system-ui, sans-serif" font-size="13">k units · ×σ</text>

            <circle cx="492" cy="80" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <circle cx="492" cy="114" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <circle cx="492" cy="148" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <text x="492" y="174" text-anchor="middle" fill="#6b6b6b"
              font-family="IBM Plex Mono, monospace" font-size="14">⋮</text>
            <circle cx="492" cy="196" r="11" fill="#fff" stroke="#0072b2" stroke-width="2"/>
            <text x="492" y="238" text-anchor="middle" fill="#2d2d2d"
              font-family="DM Sans, system-ui, sans-serif" font-size="13">n units</text>

            <g stroke="#9aabb8" stroke-width="1.15" fill="none" opacity="0.85">
              <path d="M79 80 C150 80, 190 104, 266 104"/>
              <path d="M79 114 C150 114, 190 148, 266 148"/>
              <path d="M79 148 C150 148, 190 104, 266 104"/>
              <path d="M79 148 C150 148, 190 192, 266 192"/>
              <path d="M79 196 C150 196, 190 148, 266 148"/>
              <path d="M79 196 C150 196, 190 192, 266 192"/>
              <path d="M294 104 C360 104, 400 80, 481 80"/>
              <path d="M294 104 C360 104, 400 148, 481 148"/>
              <path d="M294 148 C360 148, 400 114, 481 114"/>
              <path d="M294 148 C360 148, 400 196, 481 196"/>
              <path d="M294 192 C360 192, 400 148, 481 148"/>
              <path d="M294 192 C360 192, 400 196, 481 196"/>
            </g>

            <line x1="118" y1="136" x2="208" y2="136" stroke="#6b6b6b" stroke-width="1.4" marker-end="url(#nnArrow)"/>
            <rect x="136" y="114" width="54" height="22" rx="4" fill="#fff" stroke="#c5d4e0"/>
            <text x="163" y="129" text-anchor="middle" fill="#0072b2"
              font-family="IBM Plex Mono, monospace" font-size="12" font-weight="600">Vᵀ</text>

            <line x1="336" y1="136" x2="426" y2="136" stroke="#6b6b6b" stroke-width="1.4" marker-end="url(#nnArrow)"/>
            <rect x="354" y="114" width="54" height="22" rx="4" fill="#fff" stroke="#e0d0a8"/>
            <text x="381" y="129" text-anchor="middle" fill="#b77900"
              font-family="IBM Plex Mono, monospace" font-size="12" font-weight="600">U</text>

            <text x="556" y="148" text-anchor="start" fill="#6b6b6b"
              font-family="DM Sans, system-ui, sans-serif" font-size="12">vs Ax</text>
            <text x="556" y="168" text-anchor="start" fill="#2d2d2d"
              font-family="IBM Plex Mono, monospace" font-size="12" font-weight="500">MSE</text>
          </svg>
          <figcaption>
            Same map as $\\hat A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$, drawn as a linear
            network: $x\\mapsto V^{\\top}x$, scale by $\\sigma$, then $U$.
            The loss is mean squared error between $\\hat A$ and $A$ (equivalently between
            $\\hat A x$ and $Ax$ for all $x$).
          </figcaption>
        </figure>
      </li>
      <li>
        <p>
          A natural next idea is a
          <a href="https://en.wikipedia.org/wiki/Penalty_method" target="_blank" rel="noopener noreferrer">penalty method</a>:
          keep reconstruction, and soft-punish non-orthonormal columns so
          $U^{\\top}U \\approx I_k$ and $V^{\\top}V \\approx I_k$:
        </p>
        <div class="math">
          $$L_{\\lambda}
            = L_{\\mathrm{recon}}
            + \\lambda\\Big(
              \\|U^{\\top}U - I_k\\|_F^{2} + \\|V^{\\top}V - I_k\\|_F^{2}
            \\Big)$$
        </div>
        <p class="theory-note">
          Intuition: raise $\\lambda$ and the factors should become orthonormal; lower it and
          fitting $A$ wins. Ordinary gradient descent, no special geometry — appealing, and
          almost enough for a quick demo.
        </p>
      </li>
      <li>
        <p>
          It is broken for recovering SVD. For any <em>finite</em> $\\lambda$, the minimizer of
          $L_{\\lambda}$ is a compromise: a little orthogonality error is traded for a little
          reconstruction gain. Critical points sit <strong>beside</strong> the constraint
          $U^{\\top}U = I_k$, not on it — so the factors are not exactly SVD factors, and
          $\\sigma$ is not exactly a singular value.
        </p>
        <p class="theory-note">
          Sending $\\lambda\\to\\infty$ (or ramping it) pushes toward orthonormality but makes the
          problem ill-conditioned, and you still only approach the constraint asymptotically.
          Soft penalties approximate a hard constraint; they do not enforce one. We will not
          use $L_{\\lambda}$ here: the goal is the truncated SVD itself, including exact
          orthonormal $U$ and $V$, not a $\\lambda$-tuned soft compromise.
        </p>
      </li>
      <li>
        <p>
          Instead, minimize $L_{\\mathrm{recon}}$ with plain SGD, then <em>retract</em> after every step:
          replace $U$ and $V$ by the $Q$ factors from thin
          <a href="https://en.wikipedia.org/wiki/QR_decomposition" target="_blank" rel="noopener noreferrer">QR</a>
          so $U^{\\top}U = I_k$ and $V^{\\top}V = I_k$ exactly. Then $\\sigma$ plays the role of
          singular values. The step size is Adam-like but global: one scalar second moment
          of mean($g^{2}$) sets a shared $\\eta_t$ for every entry — no per-parameter moment
          vectors
          (<a href="#appendix-adam">why not Adam</a>;
          <a href="#appendix-vfloor">how $v$ is floored</a>).
          The loss alone does not fix signs or column order, so for display we apply
          a definite convention
          <sup class="fn"><a href="#appendix-signs">†</a></sup>
          (same idea as the classical column): sort $\\sigma$ descending, and flip each column
          so the largest-magnitude entry of $u_j$ is nonnegative.
          (See also the appendix on
          <a href="#appendix-stiefel">Stiefel manifolds and retractions</a>.)
        </p>
      </li>
    </ol>
  </section>

  <div class="controls">
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Size <em>n</em> <strong id="sizeVal">5</strong></span>
        <input id="size" type="range" min="0" max="13" step="1" value="2" />
      </label>
      <p class="help">Side length of random square <code>A</code> (n×n). The slider is log-spaced (3…64); larger <em>n</em> is slower per step.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Rank <em>k</em> <strong id="rankVal">3</strong></span>
        <input id="rank" type="range" min="1" max="5" step="1" value="3" />
      </label>
      <p class="help">How many singular components to keep (<code>1 ≤ k ≤ n</code>). Both columns use the same <em>k</em>; lower <em>k</em> is a coarser approximation.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Base LR <strong id="lrVal">0.01</strong></span>
        <input id="lr" type="range" min="0.001" max="0.20" step="0.001" value="0.01" />
      </label>
      <p class="help">Base step size $\\eta_0$. Effective $\\eta_t = \\eta_0/(\\sqrt{v^{\\dagger}}+\\sqrt{v_{\\mathrm{fp}}})$ with $v^{\\dagger}=\\max(\\hat v,\\bar g^{2},v_{\\mathrm{fp}})$ — one shared second moment; see <a href="#appendix-vfloor">Flooring the second moment</a>.</p>
    </div>
    <div class="control-row">
      <label class="slider">
        <span class="slider-label">Steps / frame <strong id="speedVal">1</strong></span>
        <input id="speed" type="range" min="1" max="20" step="1" value="1" />
      </label>
      <p class="help">SGD+QR updates before each redraw. Keep at <strong>1</strong> to watch; raise to speed up.</p>
    </div>
    <div class="control-row">
      <label class="device">
        <span class="slider-label">Device</span>
        <select id="device">
          <option value="gpu">GPU (GPU.js / WebGL)</option>
          <option value="cpu">CPU</option>
        </select>
      </label>
      <p class="help">GPU.js over WebGL (not WebGPU). CPU is often faster for tiny matrices; falls back if WebGL fails.</p>
    </div>
    <div class="control-actions">
      <div class="btns">
        <button id="play" type="button">Pause</button>
        <button id="reset" class="secondary" type="button">Reset</button>
      </div>
      <p class="help">Pause freezes training. Reset draws a new <code>A</code>, recomputes SVD, and re-inits GD.</p>
      <div class="status" id="status">step 0</div>
    </div>
  </div>

  <div class="shared">
    <div class="panel">
      <h2>Matrix A</h2>
      <canvas id="A" width="160" height="160"></canvas>
    </div>
    <div class="panel loss-wrap" style="flex:1;min-width:240px">
      <h2>Reconstruction error <span style="font-weight:500;text-transform:none;letter-spacing:0;color:inherit;opacity:0.7">(log scale)</span></h2>
      <canvas id="loss" width="520" height="140"></canvas>
      <div class="legend">
        <span><i style="background:#0072B2"></i>‖A − Â<sub>gd</sub>‖²</span>
        <span><i class="dash" style="background:#4C4C4C"></i>‖A − Â<sub>svd</sub>‖²</span>
        <span><i style="background:#E69F00"></i>‖Â<sub>svd</sub> − Â<sub>gd</sub>‖²</span>
      </div>
      <p class="chart-note">
        Closest float64 can get ≈ <a href="#appendix-fp" id="fpNote">—</a>
      </p>
    </div>
  </div>

  <div class="panel sigma-table-wrap">
    <h2>Singular values</h2>
    <div class="sigma-table-scroll">
      <table class="sigma-table" id="sigmaTable" aria-label="Singular values SVD vs GD">
        <thead>
          <tr><th scope="col"></th></tr>
        </thead>
        <tbody>
          <tr><th scope="row">SVD</th></tr>
          <tr><th scope="row">GD</th></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel" id="svdCol">
      <h2>Classical SVD (rank k)</h2>
      <div class="factor-row">
        <div class="factor"><span>Û</span><canvas id="svdU" width="100" height="100"></canvas></div>
        <div class="factor"><span>σ</span><canvas id="svdS" width="100" height="80"></canvas></div>
        <div class="factor"><span>V̂</span><canvas id="svdV" width="100" height="100"></canvas></div>
        <div class="factor"><span>Û diag(σ) V̂ᵀ</span><canvas id="svdRecon" width="120" height="120"></canvas></div>
      </div>
      <p class="note" id="svdErr"></p>
    </div>
    <div class="panel" id="gdCol">
      <h2>SGD + QR retract</h2>
      <div class="factor-row">
        <div class="factor"><span>U</span><canvas id="gdU" width="100" height="100"></canvas></div>
        <div class="factor"><span>σ</span><canvas id="gdS" width="100" height="80"></canvas></div>
        <div class="factor"><span>V</span><canvas id="gdV" width="100" height="100"></canvas></div>
        <div class="factor"><span>U diag(σ) Vᵀ</span><canvas id="gdRecon" width="120" height="120"></canvas></div>
      </div>
      <p class="note" id="gdErr"></p>
    </div>
  </div>

  <p class="note">
    <sup class="fn">†</sup>
    Why signs and column order need a convention:
    <a href="#appendix-signs">see the appendix</a>.
    GPU mode uses GPU.js over WebGL (not WebGPU). Small matrices often train faster on CPU.
  </p>

  <section class="appendix" id="appendix-stiefel" aria-label="Appendix: Stiefel manifolds and retractions">
    <h2>Appendix: Stiefel manifolds and retractions</h2>

    <h3>Stiefel manifold</h3>
    <p>
      The set of $n\\times k$ matrices with orthonormal columns is the (compact)
      <a href="https://en.wikipedia.org/wiki/Stiefel_manifold" target="_blank" rel="noopener noreferrer">Stiefel manifold</a>
    </p>
    <div class="math">
      $$\\mathrm{St}(n,k) = \\{\\, X\\in\\mathbb{R}^{n\\times k}\\,:\\, X^{\\top}X = I_k \\,\\}$$
    </div>
    <p>
      That is exactly the hard constraint on the SVD factors $U$ and $V$.
      When $k=1$ it is the unit sphere; when $k=n$ it is the orthogonal group $O(n)$.
      A plain SGD step on $U$ moves in ambient $\\mathbb{R}^{n\\times k}$ and generally
      <em>leaves</em> $\\mathrm{St}(n,k)$.
    </p>

    <h3>Retraction</h3>
    <p>
      The ideal “move along a geodesic” map is the
      <a href="https://en.wikipedia.org/wiki/Exponential_map_(Riemannian_geometry)" target="_blank" rel="noopener noreferrer">Riemannian exponential map</a>,
      which is expensive. A <em>retraction</em> is a cheaper map $R_X(\\xi)$ that
      (i) starts at $X$, (ii) agrees with a tangent step $\\xi$ to first order, and
      (iii) lands back on the manifold. Intuitively: take a step in ambient space, then snap
      back onto $\\mathrm{St}(n,k)$.
    </p>
    <p>
      On Stiefel, thin
      <a href="https://en.wikipedia.org/wiki/QR_decomposition" target="_blank" rel="noopener noreferrer">QR</a>
      is a standard retraction: after SGD updates $U$ and $V$, replace each by its $Q$ factor
      (equivalently
      <a href="https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process" target="_blank" rel="noopener noreferrer">Gram–Schmidt</a>).
      Then $U^{\\top}U = I_k$ and $V^{\\top}V = I_k$ by construction, with no $\\lambda$ to tune.
    </p>
    <div class="math">
      $$U \\leftarrow \\mathrm{qf}(U),\\qquad V \\leftarrow \\mathrm{qf}(V)$$
    </div>
    <p class="theory-note">
      Here $\\mathrm{qf}(\\cdot)$ is the $Q$ factor of a thin QR. This demo uses the Euclidean
      gradient of reconstruction plus QR. Projecting the gradient onto the Stiefel tangent
      space first is the Riemannian version; even then, the $(U,\\sigma,V)$ gauge means
      $\\hat A=\\hat A_{\\mathrm{svd}}$ need not be a discrete fixed point unless you also
      freeze or gauge-fix the factors.
    </p>

    <h3>Further reading</h3>
    <ul>
      <li>
        <a href="https://en.wikipedia.org/wiki/Penalty_method" target="_blank" rel="noopener noreferrer">Penalty method</a>
        — the soft-constraint idea in the main text.
      </li>
      <li>
        Absil, Mahony &amp; Sepulchre,
        <a href="https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds" target="_blank" rel="noopener noreferrer"><em>Optimization Algorithms on Matrix Manifolds</em></a>
        (Princeton, 2008) — retractions and Stiefel algorithms.
      </li>
      <li>
        Absil &amp; Malick,
        <a href="https://doi.org/10.1137/100802529" target="_blank" rel="noopener noreferrer">Projection-like retractions on matrix manifolds</a>,
        <em>SIAM J. Optim.</em> 22(1), 2012
        (<a href="https://sites.uclouvain.be/absil/2010-038_retractions/retraction_25PA_UCL-INMA-2010-038-v2.pdf" target="_blank" rel="noopener noreferrer">PDF</a>).
      </li>
      <li>
        <a href="https://en.wikipedia.org/wiki/Stiefel_manifold" target="_blank" rel="noopener noreferrer">Stiefel manifold</a>,
        <a href="https://en.wikipedia.org/wiki/QR_decomposition" target="_blank" rel="noopener noreferrer">QR decomposition</a>,
        <a href="https://en.wikipedia.org/wiki/Exponential_map_(Riemannian_geometry)" target="_blank" rel="noopener noreferrer">exponential map</a>
        on Wikipedia.
      </li>
    </ul>
  </section>

  <section class="appendix" id="appendix-signs" aria-label="Appendix: Signs and column order">
    <h2>Appendix: Why reorder and flip signs?</h2>

    <h3>What the loss actually determines</h3>
    <p>
      With orthonormal columns, a truncated factorization
      $\\hat A = U\\,\\mathrm{diag}(\\sigma)\\,V^{\\top}$ is unique as a
      <em>matrix</em> $\\hat A$ (the Eckart–Young approximant), but the factors $(U,\\sigma,V)$
      are not unique as arrays of columns. Two discrete freedoms remain:
    </p>
    <ul>
      <li>
        <strong>Sign flips.</strong>
        For any column index $j$, replacing $(u_j,v_j)$ by $(-u_j,-v_j)$ leaves
        $u_j\\sigma_j v_j^{\\top}$ unchanged, so $L_{\\mathrm{recon}}$ is identical.
        QR retraction does not pick which of the two signs you land on.
      </li>
      <li>
        <strong>Column order.</strong>
        Permuting the columns of $U$ and $V$ together with the entries of $\\sigma$
        yields the same $\\hat A$. If two singular values are equal (or still crossed
        during training), any orthonormal basis of that subspace is equally good for
        the loss. Even with distinct $\\sigma$, SGD has no reason to keep columns sorted
        by size unless we force it.
      </li>
    </ul>
    <p>
      So matching the classical SVD heatmaps entrywise is not implied by
      “minimize reconstruction + stay on Stiefel.” Without an extra rule, GD and classical
      SVD can look different in $U$, $V$, and the order of $\\sigma$ while $\\hat A$ (and
      $\\|A-\\hat A\\|_F^{2}$) already agree.
    </p>

    <h3>Why that matters in this demo</h3>
    <p>
      The point of the side-by-side view is to see that SGD recovers the <em>same</em>
      truncated SVD factors, not only the same reconstruction. Side-by-side $U$ panels
      are misleading if one run’s first column is the other’s third, or if a whole column
      is negated. The chart of $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$
      can go to zero while the heatmaps still look unrelated.
    </p>
    <p>
      Classical libraries already break the ambiguity with a convention (singular values
      sorted descending; a sign rule on each singular vector). We apply the same idea when
      reading factors for heatmaps — not by copying classical $U$, and not by mutating the
      training tensors (so column reorder never churns the optimizer state).
    </p>

    <h3>The convention we use</h3>
    <p>
      On a copy of the factors, after QR retraction:
    </p>
    <ol>
      <li>
        Sort columns so that $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge \\sigma_k$
        (permute $U$, $V$, and $\\mathrm{raw}$ together).
      </li>
      <li>
        For each $j$, if the largest-magnitude entry of $u_j$ is negative, flip the signs
        of both $u_j$ and $v_j$.
      </li>
    </ol>
    <p>
      The classical column in this page uses the same sign rule on its $U$. Once training
      has driven $\\hat A_{\\mathrm{gd}}$ to $\\hat A_{\\mathrm{svd}}$, the heatmaps should
      match under that shared convention.
    </p>
    <p class="theory-note">
      This is not part of the training loss — it does not change $\\hat A$ or $L_{\\mathrm{recon}}$.
      It only chooses which of the equivalent factorizations we show in the heatmaps.
    </p>
  </section>

  <section class="appendix" id="appendix-adam" aria-label="Appendix: Why not Adam">
    <h2>Appendix: Why not Adam?</h2>

    <h3>What Adam stores</h3>
    <p>
      <a href="https://en.wikipedia.org/wiki/Adam_(optimizer)" target="_blank" rel="noopener noreferrer">Adam</a>
      keeps, for <em>each parameter entry</em> $i$, exponential moving averages of the gradient
      and of its square:
    </p>
    <div class="math">
      $$m_i \\leftarrow \\beta_1 m_i + (1-\\beta_1) g_i,\\qquad
        v_i \\leftarrow \\beta_2 v_i + (1-\\beta_2) g_i^{2},\\qquad
        \\theta_i \\leftarrow \\theta_i - \\eta\\,\\frac{\\hat m_i}{\\sqrt{\\hat v_i}+\\varepsilon}.$$
    </div>
    <p>
      The vectors $m$ and $v$ are the same shape as $U$, $\\mathrm{raw}$, and $V$. That works in
      ordinary Euclidean training, where $\\theta$ only moves by small additive updates: yesterday’s
      moment at index $i$ is still “about” today’s parameter at index $i$.
    </p>

    <h3>Why that breaks with QR retraction</h3>
    <p>
      Here, after each gradient step we <em>replace</em> $U$ and $V$ by their thin-QR $Q$ factors.
      In ambient coordinates that jump is discontinuous: the new entries need not lie near the old
      ones along the axes Adam’s buffers are indexed by. Those buffers stay tied to the pre-QR
      layout, so the next update applies a stale, misaligned $m_i/\\sqrt{v_i}$ step and the
      reconstruction loss bounces.
    </p>
    <p>
      Per-parameter momentum has the same problem: a velocity buffer is tied to ambient indices
      that the retraction just scrambled. Fixing that properly means a manifold-aware optimizer
      that <em>transports</em> momentum with the retraction — more machinery than this demo needs.
    </p>

    <h3>What we use instead: one shared second moment</h3>
    <p>
      We keep Adam’s useful idea — rescale by a second-moment estimate — but store only a
      <em>single scalar</em> $v$, the EMA of mean($g_i^{2}$) over all entries of $U$,
      $\\mathrm{raw}$, and $V$. Every parameter then shares one step size
      $\\eta_t = \\eta_0/(\\sqrt{v^{\\dagger}}+\\sqrt{v_{\\mathrm{fp}}})$.
      After QR, that scalar is still a valid summary of “how big were the gradients?” — there is
      no per-entry buffer left pointing at the wrong column.
    </p>
    <p class="theory-note">
      The definition of $v^{\\dagger}$ (floors on today’s $\\bar g^{2}$ and a float64 noise level)
      is a separate issue from “per-entry vs global.” See
      <a href="#appendix-vfloor">Flooring the second moment</a>.
    </p>
  </section>

  <section class="appendix" id="appendix-vfloor" aria-label="Appendix: Flooring the second moment">
    <h2>Appendix: Flooring the second moment</h2>

    <h3>The update</h3>
    <p>
      With a shared second moment, write $\\bar g^{2}$ for the mean of $g_i^{2}$ over all trainable
      entries and
    </p>
    <div class="math">
      $$v \\leftarrow \\beta_2 v + (1-\\beta_2)\\bar g^{2},\\qquad
        \\hat v = \\frac{v}{1-\\beta_2^{t}},\\qquad
        v^{\\dagger} = \\max(\\hat v,\\,\\bar g^{2},\\,v_{\\mathrm{fp}}),\\qquad
        \\eta_t = \\frac{\\eta_0}{\\sqrt{v^{\\dagger}}+\\sqrt{v_{\\mathrm{fp}}}},\\qquad
        \\theta \\leftarrow \\theta - \\eta_t\\,\\nabla L.$$
    </div>
    <p>
      The base rate $\\eta_0$ is the slider; the status line shows the live $\\eta_t$.
      Why not just use $\\hat v$ in the denominator? Two failure modes, and neither is fixed by
      “regularizing” with $\\eta_t\\le\\eta_0$. After the Euclidean step we still QR-retract, and we
      may shrink the step further — see
      <a href="#appendix-vfloor-armijo">Armijo line search on the retract</a>.
    </p>

    <h3>Why not clip $\\eta_t\\le\\eta_0$</h3>
    <p>
      A hard cap $\\eta_t\\le\\eta_0$ looks like a simple cure for late ringing, but it is the wrong
      knob. The natural Adam-style step size (in RMS) is about
      $\\eta_0\\,\\|g\\|_{\\mathrm{rms}}/\\sqrt{\\bar g^{2}}\\approx\\eta_0$ whenever the second
      moment matches today’s gradient energy. That identity needs $\\eta_t\\approx\\eta_0/\\sqrt{\\bar g^{2}}$,
      which is <em>larger</em> than $\\eta_0$ as soon as $\\sqrt{\\bar g^{2}}&lt;1$ — the usual regime
      once MSE gradients shrink. Forcing $\\eta_t\\le\\eta_0$ then makes
      $\\|\\Delta\\|_{\\mathrm{rms}}\\approx\\eta_0\\sqrt{\\bar g^{2}}\\ll\\eta_0$, and training stalls.
      So the floor belongs on $v$, not as an artificial ceiling on $\\eta_t$.
    </p>

    <h3>Floor by today’s $\\bar g^{2}$</h3>
    <p>
      The EMA $\\hat v$ can lag <em>below</em> the current $\\bar g^{2}$: after a long stretch of
      tiny gradients, $\\beta_2\\approx 0.999$ still remembers a dying $v$, then a noisier step
      (or a jump from QR) arrives with larger $g$. Using that stale small $\\hat v$ in
      $\\eta_t=\\eta_0/(\\sqrt{\\hat v}+\\cdots)$ invents a huge learning rate and the loss rings.
    </p>

    <h3>Why no floor $\\Rightarrow$ ringing</h3>
    <p>
      Near a minimum the true gradient is small, so $\\hat v$ keeps decaying. Without the
      $\\max(\\cdot,\\bar g^{2})$ floor, nothing stops $\\sqrt{\\hat v}$ from falling far below
      $\\|g\\|_{\\mathrm{rms}}$ on the next step. The update length is then
    </p>
    <div class="math">
      $$\\|\\Delta\\|_{\\mathrm{rms}}
        = \\frac{\\eta_0}{\\sqrt{\\hat v}}\\,\\|g\\|_{\\mathrm{rms}}
        \\gg \\eta_0
        \\qquad\\text{when }\\sqrt{\\hat v}\\ll\\|g\\|_{\\mathrm{rms}}.$$
    </div>
    <p>
      That overshoot carries $\\hat A_{\\mathrm{gd}}$ away from the truncated-SVD matrix
      $\\hat A_{\\mathrm{svd}}$. The next gradient points roughly the other way, $\\hat v$ is still
      tiny, and the same oversized step fires again. On the live chart the amber curve
      $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$ oscillates; flooring with today’s
      $\\bar g^{2}$ caps $\\|g\\|_{\\mathrm{rms}}/\\sqrt{v^{\\dagger}}$ at about $1$, so that bounce
      never gets armed.
    </p>
    <figure class="ring-figure">
      <img src="${import.meta.env.BASE_URL}ringing-floor.svg" width="560" height="220"
        alt="Measured log ‖Â_svd − Â_gd‖²: floored global-RMS settles; unfloored rings" />
      <figcaption>
        Measured Node run (<code>npm run gen:ringing</code>, also a build prerequisite): same
        $A$, init, and base $\\eta_0$; blue uses floored $v^{\\dagger}$ plus Armijo on the QR
        retract, amber uses unfloored $\\eta_t=\\eta_0/(\\sqrt{\\hat v}+10^{-8})$ with no line
        search. Both curves are $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$
        (log scale). The build <em>fails</em> if amber does not ring, or if blue does not stay
        near the Eckart–Young gap.
      </figcaption>
    </figure>

    <p>
      Requiring $v^{\\dagger}\\ge\\bar g^{2}$ means we never precondition with a second moment
      smaller than the energy of <em>this</em> gradient. Then
    </p>
    <div class="math">
      $$\\|\\Delta\\|_{\\mathrm{rms}}
        = \\eta_t\\,\\|g\\|_{\\mathrm{rms}}
        \\lesssim \\frac{\\eta_0}{\\sqrt{\\bar g^{2}}}\\,\\sqrt{\\bar g^{2}}
        = \\eta_0.$$
    </div>
    <p>
      without forbidding $\\eta_t&gt;\\eta_0$. The EMA may still <em>raise</em> the denominator
      (shrink steps) when the recent past was louder than today; it just cannot quietly
      understate today’s scale.
    </p>

    <h3>Floor by float64 ULP</h3>
    <p>
      The third argument uses the same absolute scale as
      <a href="#appendix-fp">Closest you can get in floating point</a>.
      With machine epsilon $\\varepsilon$ and $\\|A\\|_{\\infty}=\\max|A_{ij}|$,
    </p>
    <div class="math">
      $$v_{\\mathrm{fp}} = \\bigl(\\varepsilon\\|A\\|_{\\infty}\\bigr)^{2}.$$
    </div>
    <p>
      A per-entry residual smaller than one ULP of $A$ is not a trustworthy signal in float64;
      gradient energy below $v_{\\mathrm{fp}}$ is in the same noise bucket. We stop letting
      $\\hat v$ fall through that floor, and we add $\\sqrt{v_{\\mathrm{fp}}}$ in the denominator
      so $\\eta_t$ stays finite if every estimate hits the floor at once.
    </p>
    <p class="theory-note">
      This is not weight decay or an $\\eta$ schedule. It is a statement about which second-moment
      values are still meaningful: at least today’s $\\bar g^{2}$, and never below float noise.
    </p>

    <h3 id="appendix-vfloor-armijo">Armijo line search on the retract</h3>
    <p>
      Flooring $v$ stops $\\eta$-blow-up ringing, but it does not make each QR step decrease
      the reconstruction loss. The update is Euclidean descent followed by thin QR; that
      retraction can move $\\hat A$ enough that the loss goes up even when $\\nabla L$ pointed
      downhill. Empirically, a floored run can reach the truncated-SVD reconstruction
      (amber gap near zero) and then drift to a worse critical point.
    </p>
    <p>
      After $\\eta_t$ is fixed, we therefore
      <a href="https://en.wikipedia.org/wiki/Backtracking_line_search" target="_blank" rel="noopener noreferrer">backtrack</a>
      $\\alpha\\in\\{1,\\tfrac12,\\tfrac14,\\ldots\\}$ on the retracted trial point
    </p>
    <div class="math">
      $$\\theta(\\alpha)=\\mathrm{QR}\\bigl(\\theta-\\alpha\\eta_t\\,\\nabla L\\bigr)$$
    </div>
    <p>
      and accept the largest $\\alpha$ that satisfies the
      <a href="https://en.wikipedia.org/wiki/Wolfe_conditions#Armijo_rule_and_curvature_condition" target="_blank" rel="noopener noreferrer">Armijo</a>
      sufficient-decrease test on the MSE,
    </p>
    <div class="math">
      $$L\\bigl(\\theta(\\alpha)\\bigr)
        \\le L(\\theta)+c\\,\\alpha\\,(-\\eta_t\\|g\\|_2^{2}),
        \\qquad c=10^{-4}.$$
    </div>
    <p>
      If no trial meets Armijo but some $\\alpha$ still lowers the loss, we take the best
      descent trial anyway — QR often breaks the Euclidean linear model behind Armijo.
      If every trial raises the loss, we reject the step and stay put. That keeps a reached
      $\\hat A_{\\mathrm{svd}}$ from walking away on the amber gap curve.
    </p>
  </section>

  <section class="appendix" id="appendix-softplus" aria-label="Appendix: Why softplus">
    <h2>Appendix: Why softplus</h2>
    <p>
      Singular values must be nonnegative. The demo stores an unconstrained vector
      $\\mathrm{raw}$ and sets $\\sigma=\\mathrm{softplus}(\\mathrm{raw})$ with
      <a href="https://en.wikipedia.org/wiki/Softplus" target="_blank" rel="noopener noreferrer">softplus</a>
      $\\ln(1+e^{x})$. Why that map, rather than something like $\\sqrt{\\mathrm{raw}^{2}}=|\\mathrm{raw}|$?
    </p>
    <p>
      Absolute value also forces $\\sigma\\ge 0$, but it has a kink at $0$: the chain-rule factor
      $\\partial\\sigma/\\partial\\mathrm{raw}$ jumps between $-1$ and $+1$ and is undefined at the origin.
      Gradient steps then chatter when a singular value wants to sit near zero.
      Softplus is smooth everywhere; its derivative is the logistic sigmoid, always in $(0,1)$.
      Sending $\\mathrm{raw}\\to-\\infty$ shrinks $\\sigma$ toward $0$ from above without a corner.
    </p>
    <p>
      For large positive $\\mathrm{raw}$, softplus is nearly the identity ($\\sigma\\approx\\mathrm{raw}$),
      so the parameterization does not warp the scale of ordinary singular values.
      Alternatives such as $\\mathrm{raw}^{2}$ are smooth and nonnegative too, but they grow
      quadratically; $e^{\\mathrm{raw}}$ stays positive but is stiff for large values.
      Softplus is the usual compromise: positive, smooth, and roughly linear when $\\sigma$ is not tiny.
    </p>
  </section>

  <section class="appendix" id="appendix-fp" aria-label="Appendix: Closest you can get in floating point">
    <h2>Appendix: Closest you can get in floating point</h2>
    <p>
      The reconstruction chart is on a log scale, so it is tempting to read “ever smaller”
      as “ever better.” In
      <a href="https://en.wikipedia.org/wiki/Double-precision_floating-point_format" target="_blank" rel="noopener noreferrer">float64</a>,
      that stops being meaningful once errors are on the order of one
      <a href="https://en.wikipedia.org/wiki/Unit_in_the_last_place" target="_blank" rel="noopener noreferrer">unit in the last place</a>
      (ULP) in the matrix entries.
    </p>
    <p>
      A rough absolute floor for $\\|\\cdot\\|_F^{2}$ is $n^{2}\\varepsilon^{2}\\|A\\|_{\\infty}^{2}$,
      where $\\varepsilon\\approx 2^{-52}$ is machine epsilon (the ULP of $1$) and
      $\\|A\\|_{\\infty}$ is the max absolute entry. For the current $A$ that is about
      <strong id="fpFloorVal">—</strong>.
      That is roughly the closest nonzero reconstruction error you can represent: below it,
      a smaller plotted value is numerical noise, not a better fit to $A$.
      The floor is not drawn on the chart — it sits far below the axis.
      The optimizer uses the same ULP scale as $v_{\\mathrm{fp}}=(\\varepsilon\\|A\\|_{\\infty})^{2}$ when
      flooring the shared second moment; see
      <a href="#appendix-vfloor">Flooring the second moment</a>.
    </p>
  </section>
`;

const el = {
  size: app.querySelector<HTMLInputElement>("#size")!,
  sizeVal: app.querySelector<HTMLElement>("#sizeVal")!,
  rank: app.querySelector<HTMLInputElement>("#rank")!,
  rankVal: app.querySelector<HTMLElement>("#rankVal")!,
  device: app.querySelector<HTMLSelectElement>("#device")!,
  lr: app.querySelector<HTMLInputElement>("#lr")!,
  lrVal: app.querySelector<HTMLElement>("#lrVal")!,
  speed: app.querySelector<HTMLInputElement>("#speed")!,
  speedVal: app.querySelector<HTMLElement>("#speedVal")!,
  play: app.querySelector<HTMLButtonElement>("#play")!,
  reset: app.querySelector<HTMLButtonElement>("#reset")!,
  status: app.querySelector<HTMLDivElement>("#status")!,
  A: app.querySelector<HTMLCanvasElement>("#A")!,
  loss: app.querySelector<HTMLCanvasElement>("#loss")!,
  svdU: app.querySelector<HTMLCanvasElement>("#svdU")!,
  svdS: app.querySelector<HTMLCanvasElement>("#svdS")!,
  svdV: app.querySelector<HTMLCanvasElement>("#svdV")!,
  svdRecon: app.querySelector<HTMLCanvasElement>("#svdRecon")!,
  svdErr: app.querySelector<HTMLParagraphElement>("#svdErr")!,
  gdU: app.querySelector<HTMLCanvasElement>("#gdU")!,
  gdS: app.querySelector<HTMLCanvasElement>("#gdS")!,
  gdV: app.querySelector<HTMLCanvasElement>("#gdV")!,
  gdRecon: app.querySelector<HTMLCanvasElement>("#gdRecon")!,
  gdErr: app.querySelector<HTMLParagraphElement>("#gdErr")!,
  sigmaTable: app.querySelector<HTMLTableElement>("#sigmaTable")!,
  fpNote: app.querySelector<HTMLAnchorElement>("#fpNote")!,
  fpFloorVal: app.querySelector<HTMLElement>("#fpFloorVal")!,
};

let A: Matrix;
let svd: SvdResult;
let trainer = new SvdGradTrainer();
let gd: GradState;
let history: LossPoint[] = [];
let playing = true;
let raf = 0;
let sharedScale = 1;
let sigmaScale = 1;
let svdFloor = 0;
let svdReconMat: Matrix;
let gpuOk = false;

/** Log-spaced matrix sizes for the n slider (index → n). */
const SIZE_STOPS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64];

function sizeFromSlider(): number {
  const i = clamp(
    Math.round(Number(el.size.value) || 0),
    0,
    SIZE_STOPS.length - 1,
  );
  el.size.value = String(i);
  return SIZE_STOPS[i];
}

function syncSliderLabels(): void {
  const n = sizeFromSlider();
  el.rank.max = String(n);
  let k = clamp(Math.round(Number(el.rank.value) || 1), 1, n);
  el.rank.value = String(k);
  el.sizeVal.textContent = String(n);
  el.rankVal.textContent = String(k);
  el.lrVal.textContent = Number(el.lr.value).toFixed(3).replace(/0$/, "");
  el.speedVal.textContent = String(Math.round(Number(el.speed.value)));
}

function readControls() {
  syncSliderLabels();
  const n = sizeFromSlider();
  const k = Number(el.rank.value);
  const device = el.device.value as DeviceKind;
  const lr = Number(el.lr.value);
  const steps = Number(el.speed.value);
  return { n, k, device, lr, steps };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function reset(newA: boolean): void {
  const { n, k, device, lr } = readControls();
  if (newA) {
    A = randomNormal(n, n, 1);
  } else if (!A || A.rows !== n) {
    A = randomNormal(n, n, 1);
  }
  svd = classicalSvd(A, k);
  svdReconMat = reconstruct(svd.U, svd.sigma, svd.V);
  sharedScale = Math.max(maxAbs(A), maxAbs(svdReconMat), 1e-6);
  sigmaScale = Math.max(...svd.sigma, 1e-6);
  svdFloor = frobeniusSq(sub(A, svdReconMat));
  const ulp = Number.EPSILON * Math.max(maxAbs(A), Number.EPSILON);
  const fpFloor = A.rows * A.cols * ulp * ulp;
  const fpLabel = fpFloor.toExponential(1);
  el.fpNote.textContent = fpLabel;
  el.fpFloorVal.textContent = fpLabel;

  let useDevice: DeviceKind = device;
  if (useDevice === "gpu" && !gpuOk) {
    useDevice = "cpu";
    el.device.value = "cpu";
    el.status.textContent = "GPU unavailable — using CPU";
  }

  try {
    trainer.init(A, k, lr, useDevice);
  } catch (e) {
    console.warn("init failed on", useDevice, e);
    trainer.init(A, k, lr, "cpu");
    el.device.value = "cpu";
  }

  history = [];
  gd = trainer.snapshot(A);
  history.push(lossSample(gd));
  paintSvd();
  paintGd();
  updateStatus();
}

function lossSample(state: GradState): LossPoint {
  const gdRecon = reconstruct(state.U, state.sigma, state.V);
  return {
    step: state.step,
    recon: state.loss.recon,
    vsSvd: frobeniusSq(sub(svdReconMat, gdRecon)),
  };
}

function paintSvd(): void {
  drawHeatmap(el.A, A, sharedScale);
  drawHeatmap(el.svdU, svd.U);
  drawHeatmap(el.svdV, svd.V);
  drawSigmaBars(el.svdS, svd.sigma, sigmaScale);
  drawHeatmap(el.svdRecon, svdReconMat, sharedScale);
  el.svdErr.textContent = `‖A − Â_svd‖_F² = ${svdFloor.toExponential(3)}`;
  paintSigmaTable();
}

function paintGd(): void {
  const recon = reconstruct(gd.U, gd.sigma, gd.V);
  drawHeatmap(el.gdU, gd.U);
  drawHeatmap(el.gdV, gd.V);
  drawSigmaBars(el.gdS, gd.sigma, sigmaScale);
  drawHeatmap(el.gdRecon, recon, sharedScale);
  drawLossChart(el.loss, history, svdFloor);
  const err = frobeniusSq(sub(A, recon));
  const vs = frobeniusSq(sub(svdReconMat, recon));
  el.gdErr.textContent = `‖A − Â_gd‖_F² = ${err.toExponential(3)}  ·  ‖Â_svd − Â_gd‖_F² = ${vs.toExponential(3)}  ·  SVD ${svdFloor.toExponential(3)}`;
  paintSigmaTable();
}

function fmtSigma(x: number): string {
  if (!Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e4 || a < 1e-4)) return x.toExponential(8);
  return x.toFixed(8);
}

/** Rows = SVD / GD; columns = σ₁…σₖ. */
function paintSigmaTable(): void {
  const k = svd.sigma.length;
  const thead = el.sigmaTable.querySelector("thead")!;
  const tbody = el.sigmaTable.querySelector("tbody")!;

  const headCells = ["", ...Array.from({ length: k }, (_, j) => String(j + 1))];
  thead.innerHTML = `<tr>${headCells.map((c) => `<th scope="col">${c}</th>`).join("")}</tr>`;

  const svdCells = svd.sigma.map((s) => `<td>${fmtSigma(s)}</td>`).join("");
  const gdCells = Array.from({ length: k }, (_, j) => {
    const s = gd.sigma[j];
    return `<td>${s === undefined ? "—" : fmtSigma(s)}</td>`;
  }).join("");

  tbody.innerHTML = `
    <tr><th scope="row">SVD</th>${svdCells}</tr>
    <tr><th scope="row">GD</th>${gdCells}</tr>
  `;
}

function updateStatus(): void {
  const { steps } = readControls();
  const eta = gd.lr.toExponential(2);
  el.status.textContent = `step ${gd.step} · ${gd.device} · ${steps}/frame · η=${eta} · L=${gd.loss.recon.toExponential(3)}`;
}

function frame(): void {
  if (playing) {
    const { steps, lr } = readControls();
    trainer.setLr(lr);
    try {
      for (let i = 0; i < steps; i++) {
        gd = trainer.stepOnce();
        history.push(lossSample(gd));
      }
      paintGd();
      updateStatus();
    } catch (e) {
      console.error(e);
      playing = false;
      el.play.textContent = "Play";
      el.status.textContent = `error — paused (${String(e)})`;
    }
  }
  raf = requestAnimationFrame(frame);
}

el.play.addEventListener("click", () => {
  playing = !playing;
  el.play.textContent = playing ? "Pause" : "Play";
});

el.reset.addEventListener("click", () => reset(true));

el.size.addEventListener("input", () => {
  syncSliderLabels();
  reset(true);
});
el.rank.addEventListener("input", () => {
  syncSliderLabels();
  reset(false);
});
el.device.addEventListener("change", () => reset(false));
el.lr.addEventListener("input", () => {
  syncSliderLabels();
  const { lr } = readControls();
  trainer.setLr(lr);
});
el.speed.addEventListener("input", () => syncSliderLabels());

gpuOk = probeGpu();
if (!gpuOk) {
  el.device.value = "cpu";
}
syncSliderLabels();
reset(true);
raf = requestAnimationFrame(frame);

void window.MathJax?.typesetPromise?.([app]).catch(() => {
  const t = window.setInterval(() => {
    if (window.MathJax?.typesetPromise) {
      window.clearInterval(t);
      void window.MathJax.typesetPromise([app]);
    }
  }, 100);
  window.setTimeout(() => window.clearInterval(t), 5000);
});

void raf;
