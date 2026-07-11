# SVD via Gradient Descent

Interactive demo: recover a rank-`k` factorization by Adam on

```
L = ||A − U diag(σ) Vᵀ||_F² + λ ( ||UᵀU − I||_F² + ||VᵀV − I||_F² )
```

and compare it live to truncated classical SVD on the same matrix.

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
- **λ**, **learning rate**, **steps/frame** — loss weight, Adam step size, animation speed
- **Device** — CPU or GPU (GPU.js over **WebGL**, not WebGPU)
- Play / Pause / Reset

Expand each control’s help text beside the slider for what it does.

The GD column heatmaps animate each frame; the classical SVD column stays fixed until you change `A`, `n`, or `k`.

### Deploy note

`js-pytorch` depends on `@eduardoleao052/gpu`. For the browser, Vite resolves that package to `gpu-browser.js` (pure JS / WebGL). Native Node `gl` may sit in `node_modules` but is not what the web bundle uses. Prefer `npm install --ignore-scripts` if the native addon fails to build.

Small matrices often train faster on CPU; GPU is available for completeness.

## Python twin

Same loss against real PyTorch (CPU):

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
