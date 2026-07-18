import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: ".",
  base: process.env.VITE_BASE || "/",
  server: { port: 5173 },
  resolve: {
    alias: {
      "js-pytorch": path.resolve(root, "src/vendor/js-pytorch-browser.js"),
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(root, "index.html"),
        reflections: path.resolve(root, "reflections.html"),
        svd: path.resolve(root, "svd.html"),
        truncate: path.resolve(root, "truncate.html"),
        faces: path.resolve(root, "faces.html"),
        gradient: path.resolve(root, "gradient.html"),
        noise: path.resolve(root, "noise.html"),
      },
    },
  },
});
