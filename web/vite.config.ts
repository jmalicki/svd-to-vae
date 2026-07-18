import { existsSync, renameSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

/**
 * One repo, two sites:
 *   VITE_SITE=tour (default) — the six-chapter tour at jmalicki.github.io/svd-to-vae/
 *   VITE_SITE=grad           — the single-page SVD-via-gradient-descent demo at
 *                              jmalicki.github.io/svd-grad/ (gradient chapter as index.html)
 * CI selects the variant from the repository name; see .github/workflows/pages.yml.
 */
const site = process.env.VITE_SITE === "grad" ? "grad" : "tour";

const tourInputs = {
  main: path.resolve(root, "index.html"),
  reflections: path.resolve(root, "reflections.html"),
  svd: path.resolve(root, "svd.html"),
  truncate: path.resolve(root, "truncate.html"),
  faces: path.resolve(root, "faces.html"),
  gradient: path.resolve(root, "gradient.html"),
  noise: path.resolve(root, "noise.html"),
};

const gradInputs = {
  gradient: path.resolve(root, "gradient.html"),
};

/** Serve the gradient chapter as the site root and drop tour-only assets. */
function gradSinglePage(): Plugin {
  return {
    name: "grad-single-page",
    closeBundle() {
      const dist = path.resolve(root, "dist");
      const page = path.resolve(dist, "gradient.html");
      if (existsSync(page)) renameSync(page, path.resolve(dist, "index.html"));
      // The IMM face pack (~37 MB) belongs to the faces chapter only.
      rmSync(path.resolve(dist, "imm"), { recursive: true, force: true });
    },
  };
}

export default defineConfig({
  root: ".",
  base: process.env.VITE_BASE || "/",
  server: { port: 5173 },
  resolve: {
    alias: {
      "js-pytorch": path.resolve(root, "src/vendor/js-pytorch-browser.js"),
    },
  },
  plugins: site === "grad" ? [gradSinglePage()] : [],
  build: {
    rollupOptions: {
      input: site === "grad" ? gradInputs : tourInputs,
    },
  },
});
