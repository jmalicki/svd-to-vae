# SVD via Gradient Descent

Interactive demo: recover a rank-`k` factorization by plain SGD on

```
L = ||A − U diag(σ) Vᵀ||_F²
```

then **retract** with thin QR after every step so `UᵀU = I` and `VᵀV = I`
(Stiefel), and compare live to truncated classical SVD on the same matrix.

## Web demo

```bash
cd web
npm install --ignore-scripts   # skip native `gl` compile if it fails
npm run dev
```

The app imports a **vendored browser build** of js-pytorch (`web/src/vendor/js-pytorch-browser.js`) so Vite does not pull the Node `createRequire` entry. GPU mode uses GPU.js / WebGL from that bundle. Refresh after upgrading js-pytorch with:

```bash
cd web && bash scripts/vendor-js-pytorch.sh
```

Open the printed local URL. Controls (all numeric params are **sliders**):

- **Size n** / **Rank k** — matrix shape and factorization rank
- **Learning rate**, **steps/frame** — SGD step size and animation speed
- **Device** — CPU or GPU (GPU.js over **WebGL**, not WebGPU)
- Play / Pause / Reset

The GD column heatmaps animate each frame; the classical SVD column stays fixed until you change `A`, `n`, or `k`. The loss chart shows reconstruction vs the truncated-SVD floor (QR keeps factors orthonormal by construction).

Plain SGD is used instead of Adam: Adam’s momentum buffers fight hard QR retraction (and column reordering), which shows up as a bouncing loss.

### Deploy note

`js-pytorch` depends on `@eduardoleao052/gpu`. For the browser, Vite resolves that package to `gpu-browser.js` (pure JS / WebGL). Native Node `gl` may sit in `node_modules` but is not what the web bundle uses. Prefer `npm install --ignore-scripts` if the native addon fails to build.

Small matrices often train faster on CPU; GPU is available for completeness.

## Python twin

Same algorithm against real PyTorch (CPU):

```bash
cd python
pip install -r requirements.txt
python train.py --n 5 --rank 3 --steps 800
```

Compares final reconstruction error to `torch.linalg.svd` truncated to rank `k`.

## Notes

- Factors match SVD only up to **sign flips** and **column order**.
- Classical SVD in the browser uses [`ml-matrix`](https://github.com/mljs/matrix) (`SingularValueDecomposition`). js-pytorch has no `linalg.svd`.
- `σ = softplus(raw)` keeps singular values non-negative; column scaling `U * σ` keeps `σ` in the autograd graph.
- Retraction is Euclidean gradient of reconstruction + thin QR (modified Gram–Schmidt in the browser; `torch.linalg.qr` in Python)—not a full Riemannian tangent-space projection. Background: [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold), [QR](https://en.wikipedia.org/wiki/QR_decomposition), [exponential map](https://en.wikipedia.org/wiki/Exponential_map_(Riemannian_geometry)), Absil–Mahony–Sepulchre [*Optimization Algorithms on Matrix Manifolds*](https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds), [Absil & Malick (SIAM J. Optim. 2012)](https://doi.org/10.1137/100802529) ([PDF](https://sites.uclouvain.be/absil/2010-038_retractions/retraction_25PA_UCL-INMA-2010-038-v2.pdf)).
