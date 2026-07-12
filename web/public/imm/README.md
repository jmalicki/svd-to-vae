# IMM Face Database (full)

240 annotated images (40 subjects × 6) with ASF landmarks from the
[IMM Face Database](http://www2.imm.dtu.dk/pubdb/pubs/3160-full.html).

Free for education and research; cite:

> M. B. Stegmann, B. K. Ersbøll, and R. Larsen. FAME – a flexible appearance
> modelling environment. IEEE Trans. on Medical Imaging, 22(10):1319–1331, 2003.

Mirrored via OpenIMAJ (`http://datasets.openimaj.org/imm_face_db.zip`).

## Prebaked pack

`npm run gen:imm-pack` (also `predev` / `prebuild`) runs the same JS/TS warp
and SVD pipeline the browser could run, decoding JPEGs with pure-JS `jpeg-js`,
and writes:

- `examples.bin` — warped 64×64 appearances + thumbs + landmarks
- `model.bin` — appearance/shape SVD factors + pixel-foil SVD

Pages load those two files instead of fetching 240 jpg+asf pairs or computing
SVD on the main thread.

`npm run test:imm-pack` (also at the end of `gen:imm-pack`, and in CI) times
unpacking both packs and **fails if unpack exceeds 3s** or full-rank
reconstruction of a training face is bad — so a regression that brings back
runtime SVD cannot slip through quietly.
