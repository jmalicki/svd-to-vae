import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: ".",
  server: { port: 5173 },
  resolve: {
    alias: {
      // npm ESM entry pulls Node createRequire/fs — use browser build instead.
      "js-pytorch": path.resolve(root, "src/vendor/js-pytorch-browser.js"),
    },
  },
});
