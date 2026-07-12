# IMM Face Database (full)

240 annotated images (40 subjects × 6) with ASF landmarks from the
[IMM Face Database](http://www2.imm.dtu.dk/pubdb/pubs/3160-full.html).

Free for education and research; cite:

> M. B. Stegmann, B. K. Ersbøll, and R. Larsen. FAME – a flexible appearance
> modelling environment. IEEE Trans. on Medical Imaging, 22(10):1319–1331, 2003.

Mirrored via OpenIMAJ (`http://datasets.openimaj.org/imm_face_db.zip`).

## Prebaked pack

`npm run gen:imm-pack` (also `predev` / `prebuild`) runs the same JS/TS warp
pipeline the browser could run (`buildExamplesFromLoaded` + `encodeImmPack`),
decoding JPEGs with pure-JS `jpeg-js`, and writes `examples.bin`. Pages load
that single file instead of 240 jpg+asf pairs.
