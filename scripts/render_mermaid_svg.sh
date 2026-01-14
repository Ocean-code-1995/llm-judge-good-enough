#!/usr/bin/env bash
set -euo pipefail

# Render a Mermaid .mmd file to SVG with a transparent background.
#
# Usage:
#   ./scripts/render_mermaid_svg.sh
#   ./scripts/render_mermaid_svg.sh -i diagrams/mermaid/general_approach.mmd -o diagrams/svg/general_approach.svg
#
# Requirements (either one):
#   - Preferred: `mmdc` from @mermaid-js/mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
#   - Fallback:  `npx` (ships with Node) to run @mermaid-js/mermaid-cli on-demand

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IN="${REPO_ROOT}/diagrams/mermaid/general_approach.mmd"
OUT="${REPO_ROOT}/diagrams/svg/general_approach.svg"
BACKGROUND="transparent"

usage() {
  cat <<EOF
Render Mermaid .mmd -> .svg (transparent background).

Usage:
  $(basename "$0") [-i input.mmd] [-o output.svg] [-b background]

Defaults:
  -i ${IN#"$REPO_ROOT"/}
  -o ${OUT#"$REPO_ROOT"/}
  -b ${BACKGROUND}

Examples:
  $(basename "$0")
  $(basename "$0") -i diagrams/mermaid/general_approach.mmd -o diagrams/svg/general_approach.svg
EOF
}

while getopts ":i:o:b:h" opt; do
  case "$opt" in
    i) IN="${REPO_ROOT}/${OPTARG#./}" ;;
    o) OUT="${REPO_ROOT}/${OPTARG#./}" ;;
    b) BACKGROUND="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Missing argument for -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$IN" ]]; then
  echo "❌ Input file not found: $IN" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "🧩 Input : ${IN#"$REPO_ROOT"/}"
echo "🖼️  Output: ${OUT#"$REPO_ROOT"/}"
echo "🎨 Background: ${BACKGROUND}"

if command -v mmdc >/dev/null 2>&1; then
  echo "✅ Using installed mermaid-cli (mmdc)"
  mmdc -i "$IN" -o "$OUT" -b "$BACKGROUND"
else
  if ! command -v npx >/dev/null 2>&1; then
    echo "❌ Neither 'mmdc' nor 'npx' found." >&2
    echo "   Install one of:" >&2
    echo "   - npm install -g @mermaid-js/mermaid-cli" >&2
    echo "   - install Node.js (for npx)" >&2
    exit 1
  fi
  echo "✅ Using npx fallback (@mermaid-js/mermaid-cli)"
  npx -y @mermaid-js/mermaid-cli -i "$IN" -o "$OUT" -b "$BACKGROUND"
fi

echo "🎉 Done: ${OUT#"$REPO_ROOT"/}"


