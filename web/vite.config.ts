import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: ".",
  // CI sets VITE_BASE=/svd-grad/ for GitHub Pages project URLs.
  base: process.env.VITE_BASE || "/",
  server: { port: 5173 },
  resolve: {
    alias: {
      // npm ESM entry pulls Node createRequire/fs — use browser build instead.
      "js-pytorch": path.resolve(root, "src/vendor/js-pytorch-browser.js"),
    },
  },
});
