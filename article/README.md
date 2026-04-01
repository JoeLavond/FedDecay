# FedDecay — Article

Paper source for *Balancing Model Performance and Rapid Personalization in Federated Learning with Learning Rate Scheduling* (FedDecay), submitted to JCGS (manuscript JCGS-25-127).

## Building

Run from this directory (`article/`). Requires `latexmk` and a standard TeX distribution (e.g. TeX Live, MacTeX).

```bash
make jcgs   # builds paper/2_main_jcgs.pdf  (JCGS submission)
make tmlr   # builds paper/1_main_tmlr.pdf  (TMLR format)
make jmlr   # builds paper/0_main_jmlr.pdf  (JMLR format)
make clean  # remove all build artifacts
```

`latexmk` handles multi-pass compilation (BibTeX, cross-references) automatically. All three targets share the same source content — only the preamble and style differ.

## Directory Structure

```
article/
├── Makefile
├── revisions.md              # Point-by-point JCGS revision notes
├── feedback/                 # Editor/reviewer PDFs and correspondence
├── images/                   # Compiled figure PNGs referenced by the paper
│   ├── decay-scheme.png
│   ├── gradient-emphasis.png
│   ├── method-costs.png
│   ├── run_costs.png
│   └── sensitivity-analysis-best-beta.png
├── paper/
│   ├── 0_main_jmlr.tex       # JMLR entry point
│   ├── 1_main_tmlr.tex       # TMLR entry point
│   ├── 2_main_jcgs.tex       # JCGS entry point (active submission)
│   ├── references.bib
│   ├── source/               # Journal style files (.sty, .bst) and math_commands.tex
│   ├── jmlr-cover-letter/    # Cover letter for JMLR submission
│   └── content/
│       ├── document/         # Main body sections (§1–§5)
│       │   ├── 1-introduction.tex
│       │   ├── 2-background.tex
│       │   ├── 3-methodology/
│       │   │   ├── 3a-introduction.tex
│       │   │   ├── 3b-updates.tex
│       │   │   └── 3c-decay.tex
│       │   ├── 4-experiment/
│       │   │   ├── 4a-setting.tex
│       │   │   ├── 4b-generalization.tex
│       │   │   └── 4c-costs.tex
│       │   └── 5-conclusion.tex
│       ├── appendix/
│       │   ├── experiment/   # App. A: sensitivity, B: linear decay, C: seed, D: hyperparams
│       │   └── methodology/  # App.: Taylor expansion proof, convergence proof
│       ├── figures/          # \begin{figure} wrappers that \includegraphics from images/
│       └── tables/           # Tabular environments for results and parameter tables
└── STOR_712_Project/         # Original course project slides and report (archival)
```

## The `\edit` Command

All three entry-point files define `\edit{...}`. In the JCGS file (`2_main_jcgs.tex`) and the TMLR file (`1_main_tmlr.tex`) this renders as **blue text**, marking content added or revised in response to reviewer feedback. To produce a clean (uncolored) version, redefine the command in the preamble:

```latex
\renewcommand\edit[1]{#1}
```

## Revision Workflow

Reviewer responses and a change log are in `revisions.md`. Each entry identifies the reviewer comment, the response, and the specific `.tex` files modified. The `feedback/` directory contains the original PDFs from the journal.
