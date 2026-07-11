import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "js-pytorch": path.resolve(root, "src/vendor/js-pytorch-browser.js"),
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
    testTimeout: 30_000,
  },
});
