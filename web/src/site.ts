/**
 * Build-time site variant. One repo builds two GitHub Pages sites:
 *   "tour" — the six-chapter SVD-to-VAE tour (jmalicki.github.io/svd-to-vae/)
 *   "grad" — the original standalone SVD-via-gradient-descent demo
 *            (jmalicki.github.io/svd-grad/), which is the gradient chapter
 *            served as index.html with chapter links pointed at the tour.
 */
export const SITE: "tour" | "grad" =
  import.meta.env.VITE_SITE === "grad" ? "grad" : "tour";

export const TOUR_URL = "https://jmalicki.github.io/svd-to-vae/";
