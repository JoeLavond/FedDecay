# FedDecay — Article

Paper source for *FedDecay: Balancing Model Performance and Rapid Personalization in Federated Learning with Learning Rate Scheduling*, accepted at the Journal of Computational and Graphical Statistics (JCGS).

## Building

Run from this directory (`article/`). Requires `latexmk` and a standard TeX distribution (e.g. TeX Live, MacTeX).

```bash
make jcgs   # builds paper/main_jcgs.pdf
make clean  # remove all build artifacts
```

`latexmk` handles multi-pass compilation (BibTeX, cross-references) automatically.

## Directory Structure

```
article/
├── Makefile
├── images/                   # Compiled figure PNGs referenced by the paper
│   ├── decay-scheme.png
│   ├── gradient-emphasis.png
│   ├── method-costs.png
│   ├── run_costs.png
│   └── sensitivity-analysis-best-beta.png
└── paper/
    ├── main_jcgs.tex       # Main entry point
    ├── references.bib
    ├── source/               # Journal style files (.sty, .bst) and math_commands.tex
    └── content/
        ├── document/         # Main body sections (§1–§5)
        │   ├── 1-introduction.tex
        │   ├── 2-background.tex
        │   ├── 3-methodology/
        │   │   ├── 3a-introduction.tex
        │   │   ├── 3b-updates.tex
        │   │   └── 3c-decay.tex
        │   ├── 4-experiment/
        │   │   ├── 4a-setting.tex
        │   │   ├── 4b-generalization.tex
        │   │   └── 4c-costs.tex
        │   └── 5-conclusion.tex
        ├── appendix/
        │   ├── experiment/   # App. A: sensitivity, B: linear decay, C: seed, D: hyperparams
        │   └── methodology/  # App.: Taylor expansion proof, convergence proof
        ├── figures/          # \begin{figure} wrappers referencing images/
        └── tables/           # Tabular environments for results and parameter tables
```
