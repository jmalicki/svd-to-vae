# SVD to VAE

Interactive tour from truncated SVD toward generative models — without naming
VAE / ELBO / KL until a later chapter.

Live: [https://jmalicki.github.io/svd-to-vae/](https://jmalicki.github.io/svd-to-vae/)
(built and deployed from `main` by GitHub Actions).

| Chapter | Page | Idea |
| --- | --- | --- |
| 1 | [`/`](https://jmalicki.github.io/svd-to-vae/) | SVD geometry: circle → ellipse, singular values |
| 2 | [`/truncate.html`](https://jmalicki.github.io/svd-to-vae/truncate.html) | Truncated SVD on a matrix |
| 3 | [`/faces.html`](https://jmalicki.github.io/svd-to-vae/faces.html) | Face compression (IMM + warp + appearance SVD) |
| 4 | [`/gradient.html`](https://jmalicki.github.io/svd-to-vae/gradient.html) | Recover factors by gradient descent |
| 5 | [`/noise.html`](https://jmalicki.github.io/svd-to-vae/noise.html) | Noise in the bottleneck codes |

The standalone GD demo that started this work still lives at
[svd-grad](https://jmalicki.github.io/svd-grad/)
([repo](https://github.com/jmalicki/svd-grad)).

## Local

```bash
cd web
npm install --ignore-scripts   # skip native `gl` compile if it fails
npm run dev
```

`predev` / `prebuild` run `npm run gen:ringing`, which trains floored global-RMS + Armijo vs unfloored global-RMS (no line search), plots $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$, writes `public/ringing-floor.svg`, and **fails the build if the unfloored curve does not ring or the floored curve fails to stay near the Eckart–Young gap**.

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
