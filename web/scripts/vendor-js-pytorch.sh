#!/usr/bin/env bash
# Refresh vendored js-pytorch browser build from node_modules.
set -euo pipefail
cd "$(dirname "$0")/.."
src="node_modules/js-pytorch/dist/js-pytorch-browser.js"
dst="src/vendor/js-pytorch-browser.js"
test -f "$src" || { echo "run npm install first"; exit 1; }
mkdir -p src/vendor
python3 - "$src" "$dst" <<'PY'
from pathlib import Path
import sys
src_path, dst_path = Path(sys.argv[1]), Path(sys.argv[2])
text = src_path.read_text()
if not text.startswith("var require;"):
    text = "var require;\n" + text
old = "exports = {}\nexports.torch = torch;"
new = "export { torch };\nglobalThis.torch = torch;"
if old in text:
    text = text.replace(old, new)
elif "export { torch }" not in text:
    text += "\nexport { torch };\nglobalThis.torch = torch;\n"
dst_path.write_text(text)
print(f"wrote {dst_path} ({dst_path.stat().st_size} bytes)")
PY
