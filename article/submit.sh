#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "==> Recompiling paper..."
make clean && make jcgs

echo "==> Creating submission zip..."
rm -f FedDecay_submission.zip
zip -r FedDecay_submission.zip \
  paper/main_jcgs.tex \
  paper/main_jcgs.bbl \
  paper/references.bib \
  paper/content/ \
  paper/source/math_commands.tex \
  paper/agsm.bst \
  images/

echo ""
echo "Done. Send the following to eric@amstat.org:"
echo "  1. paper/main_jcgs.pdf          (unblinded PDF)"
echo "  2. FedDecay_submission.zip       (TeX source + figures)"
echo "  3. Code supplement URL:          https://github.com/JoeLavond/FedDecay"
