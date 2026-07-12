---
name: interactive-math-chapter
description: >-
  Design or rewrite interactive math teaching chapters (copy, demos, viz
  sequencing) for learners who know basic linear algebra. Use when writing or
  editing educational pages, chapter copy, interactive demos, SVD-to-VAE tour
  chapters, or when the user asks for pedagogy, teaching sequence, or clearer
  math exposition.
---

# Interactive math chapter pedagogy

Apply these principles when creating or rewriting any chapter in an interactive
math tour. Audience default: knows basic linear algebra (vectors, $Ax$, matrix
multiply, maybe rotations/orthogonal). Do **not** use opaque jargon such as
“linear map,” “codomain,” or “orthonormal preamble” unless defined in place
with a concrete picture.

## Non-negotiables

1. **One idea per section.** Headline + short prose + one demo or figure that
   does that job. No dashboard of competing widgets in the first viewport.
2. **Analysis before synthesis.** Show a concrete object acting (matrix on a
   circle, compression of a face, …) *before* controls that rebuild the object
   from abstract factors.
3. **See / measure, then name.** Introduce Greek letters and formal names only
   after the learner has seen the quantity (axis length, error, code width).
4. **Geometry before algebra for orthogonal maps.** Householder = mirror first;
   $I-2nn^{\\top}$ only after the picture. Aim-onto-axis via perpendicular
   bisector before “introduce a zero.” Do not present classical Gram–Schmidt as
   the path to the SVD (GS/QR is a different factorization; Householder appears
   in SVD *bidiagonalization*).
5. **Avoid special cases that hide the concept.** First worked examples must
   force the distinction you care about (e.g. tilted $A$ so $\sigma$ ≠ diagonal
   entries; warped faces so pixel-SVD fails). Diagonal / identity / toy cases
   come later as simplifying checks.
6. **Show intermediate stages for multi-step stories.** If the claim is
   “rotate → stretch → rotate,” show four frames, not only before/after of the
   product. If the claim is “encode → bottleneck → decode,” show the middle.
7. **Interactive demos isolate one lever.** Each scrubber should answer one
   question the prose just posed. Dumping all parameters at once is not
   “interactive pedagogy.”
8. **Notation earns its keep.** Prefer “stretch amounts,” “how many numbers we
   keep,” “noise on the codes” until a symbol is reused enough to pay off.
9. **Bridge forward, don’t foreshadow surprises.** One bland sentence to the
   next chapter is fine. Do not name VAE / ELBO / KL (or other later punchlines)
   early unless this *is* the reveal chapter. Do not name SVD until the learner
   has seen the geometric pieces on that chapter.
10. **Acceptance test.** End a chapter rewrite only when a bright learner can
    state 2–4 concrete takeaways in plain language (write them in the PR/commit
    notes or at the bottom of the chapter plan).

## Recommended page skeleton

Use this order unless the chapter’s idea truly requires otherwise:

1. **Hook** — one concrete fact or picture (no formula dump).
2. **Short setup** — what “before” and “after” mean for this page.
3. **Worked example** — fixed, non-degenerate instance with numbers and a figure.
4. **Analysis playground** — edit the natural object; watch the phenomenon.
5. **Mechanism / movie** — intermediate stages; then the compact formula if needed.
6. **Synthesis playground** (optional) — rebuild from pieces the learner now knows.
7. **Appendix** — vocabulary and edge cases; not required reading to get the idea.
8. **Next** — bland link; no spoiler.

## Language

| Prefer | Avoid (unless defined with a picture) |
| --- | --- |
| multiply by $A$, where $(1,0)$ goes | linear map, transformation of space |
| stretch / squash | spectral / extremal (unexplained) |
| rotate / twist the axes | change of orthonormal frame |
| how many numbers we keep ($k$) | latent dimensionality (early chapters) |
| jiggle the codes | inject isotropic Gaussian noise (early) |

Basic LA vocabulary is fine: vector, matrix, $Ax$, transpose, orthogonal,
eigenvalue *when this chapter is about eigenvalues*.

## Interactive design checklist

Copy and track:

```
Chapter pedagogy:
- [ ] Hook has one concrete claim
- [ ] First example is non-degenerate for the concept
- [ ] Names/symbols appear after a visible quantity
- [ ] Multi-step claim has intermediate frames
- [ ] Primary demo is analysis-first
- [ ] Each control maps to one prose question
- [ ] No early spoilers for later chapters
- [ ] 2–4 plain-language takeaways written down
```

## Anti-patterns (fix these)

- Leading with $A = U\Sigma V^{\top}$ (or any factorization) before the movie.
- Diagonal-only first example when the general case is tilted / coupled.
- Heatmaps, factor bars, and geometry all competing before geometry is clear.
- Synthesis sliders ($\sigma$, angles, $k$, $\tau$) as the *first* interaction.
- Calling something by its research name in the lede (“ELBO”, “Stiefel”, “Procrustes”)
  without a plain-language sentence first.
- Duplicate demos that repeat the same picture without a harder second question.

## When reviewing an existing chapter

1. State the intended takeaways in one bullet list.
2. Mark where names appear relative to first sight of the quantity.
3. Check the first example for degeneracy.
4. Check whether the primary demo is analysis or synthesis.
5. Propose a reorder before proposing new widgets.
