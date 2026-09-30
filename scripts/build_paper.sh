#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error "$project_dir/paper/main.tex"
gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
  -dEmbedAllFonts=true -dSubsetFonts=true \
  -dDownsampleColorImages=true -dColorImageResolution=180 \
  -dDownsampleGrayImages=true -dGrayImageResolution=180 \
  -dDownsampleMonoImages=true -dMonoImageResolution=600 \
  -sOutputFile="$project_dir/paper/submission.pdf" "$project_dir/paper/main.pdf"
