# SVD to VAE

Interactive tour from truncated SVD toward generative models — without naming
VAE / ELBO / KL until a later chapter.

Live: [https://jmalicki.github.io/svd-to-vae/](https://jmalicki.github.io/svd-to-vae/)
(built and deployed from `main` by GitHub Actions).

| Chapter | Page | Idea |
| --- | --- | --- |
| 1 | [`/`](https://jmalicki.github.io/svd-to-vae/) | Matrix as linear transformation: rotate, stretch |
| 2 | [`/svd.html`](https://jmalicki.github.io/svd-to-vae/svd.html) | Householder mirrors → zeros → stretches; then name the SVD |
| 3 | [`/truncate.html`](https://jmalicki.github.io/svd-to-vae/truncate.html) | Truncated SVD on a matrix |
| 4 | [`/faces.html`](https://jmalicki.github.io/svd-to-vae/faces.html) | Face compression (IMM + warp + appearance SVD) |
| 5 | [`/gradient.html`](https://jmalicki.github.io/svd-to-vae/gradient.html) | Recover factors by gradient descent |
| 6 | [`/noise.html`](https://jmalicki.github.io/svd-to-vae/noise.html) | Noise in the bottleneck codes |

The standalone GD demo that started this work still lives at
[svd-grad](https://jmalicki.github.io/svd-grad/)
([repo](https://github.com/jmalicki/svd-grad)).

## Local

```bash
cd web
npm install --ignore-scripts   # skip native `gl` compile if it fails
npm run dev
```

`predev` / `prebuild` run `npm run gen:ringing`, which trains floored global-RMS + Armijo vs unfloored global-RMS (no line search), plots $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$, writes `public/ringing-floor.svg`, and **fails the build if the unfloored curve does not ring or the floored curve fails to stay near the Eckart–Young gap**. They also run `npm run gen:imm-pack`, which warps the IMM faces and bakes SVD factors into `examples.bin` / `model.bin`, then **`npm run test:imm-pack` fails if unpacking those packs takes more than 3s** (guards against bringing runtime SVD back onto the page-load path). CI runs the same generators and timing check before `vite build`.

```bash
cd web && npm test
```

The app imports a **vendored browser build** of js-pytorch (`web/src/vendor/js-pytorch-browser.js`). Refresh after upgrading js-pytorch with:

```bash
cd web && bash scripts/vendor-js-pytorch.sh
```

## Face data

Chapters 2 and 4 use the [IMM Face Database](https://www2.imm.dtu.dk/~aam/datasets/datasets.html)
(Stegmann / FAME; education & research). Cite accordingly. Bundled under
`web/public/imm/` with landmarks (`.asf`) and images.

## Python twin

Same GD algorithm against real PyTorch (CPU), from the gradient chapter:

```bash
cd python
pip install -r requirements.txt
python train.py --n 5 --rank 3 --steps 800
```

## License

[MIT](LICENSE)
