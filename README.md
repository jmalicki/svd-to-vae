# SVD to VAE

Interactive tour from truncated SVD toward generative models — without naming
VAE / ELBO / KL until a later chapter.

Live: [https://jmalicki.github.io/svd-to-vae/](https://jmalicki.github.io/svd-to-vae/)
(built and deployed from `main` by GitHub Actions).

| Chapter | Page | Idea |
| --- | --- | --- |
| 1 | [`/`](https://jmalicki.github.io/svd-to-vae/) | Matrix as linear transformation: rotate, stretch |
| 2 | [`/reflections.html`](https://jmalicki.github.io/svd-to-vae/reflections.html) | Householder mirrors → zeros → stretches |
| 3 | [`/svd.html`](https://jmalicki.github.io/svd-to-vae/svd.html) | Singular value decomposition and truncation |
| 4 | [`/faces.html`](https://jmalicki.github.io/svd-to-vae/faces.html) | Face compression (IMM + warp + appearance SVD) |
| 5 | [`/gradient.html`](https://jmalicki.github.io/svd-to-vae/gradient.html) | Recover factors by gradient descent |
| 6 | [`/noise.html`](https://jmalicki.github.io/svd-to-vae/noise.html) | Noise in the bottleneck codes |

## Two sites, one repo

The standalone GD demo that started this work still lives at
[svd-grad](https://jmalicki.github.io/svd-grad/)
([repo](https://github.com/jmalicki/svd-grad)) and is built **from this same
codebase**: the same commit is pushed to both `jmalicki/svd-to-vae` and
`jmalicki/svd-grad`, and each repo's Pages workflow builds its own variant.

- `VITE_SITE=tour` (default) builds all six chapters — deployed to
  `jmalicki.github.io/svd-to-vae/`.
- `VITE_SITE=grad` builds only the gradient chapter, serves it as
  `index.html`, drops the 37 MB IMM face pack, and points its chapter links
  at the tour — deployed to `jmalicki.github.io/svd-grad/`.

The workflow picks the variant from the repository name
(`.github/workflows/pages.yml`), so keeping both sites fresh is just:

```bash
git push          # origin is configured with both push URLs
```

## Local

```bash
cd web
npm install --ignore-scripts   # skip native `gl` compile if it fails
npm run dev
```

`predev` / `prebuild` run `npm run gen:ringing`, which trains a 2×2 ablation of global-RMS SGD ({v† floor on/off} × {Armijo on/off}), plots $\\|\\hat A_{\\mathrm{svd}}-\\hat A_{\\mathrm{gd}}\\|_F^{2}$ for all four runs, writes `public/ringing-floor.svg`, and **fails the build if the no-safeguard curve does not ring or the floor+Armijo curve fails to stay near the Eckart–Young gap**. They also run `npm run gen:imm-pack`, which warps the IMM faces and bakes SVD factors into `examples.bin` / `model.bin`, then **`npm run test:imm-pack` fails if unpacking those packs takes more than 3s** (guards against bringing runtime SVD back onto the page-load path). CI runs the same generators and timing check before `vite build`.

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
