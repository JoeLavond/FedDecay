# FedDecay — Revision Notes (JCGS-25-127)

Response to the Editor (Yuguo Chen), Associate Editor, Reviewer 1, and Reviewer 2.
All section and file references below are to the revised manuscript.

---

## Editor Requirements

### E1 — Manuscript length
**Feedback:** The revision should not increase the length of the manuscript.

**Response:** All additions were offset by removing or condensing redundant and self-limiting phrasing in the same sections. Net additions are a small number of sentences distributed across the methodology, experiments, and appendix. No new sections or subsections were added.

---

### E2 — Code documentation
**Feedback:** Code should be documented, user-friendly, and runnable in a clean environment.

**Response:** Four files in the repository were updated to address documentation and reproducibility. All changes are in the `bench` branch at `https://github.com/JoeLavond/FedDecay/tree/bench`.

**Changes made:**

1. `README.md` — Replaced the minimal stub with a complete Getting Started guide. The new README covers: cloning the repository; creating a fresh conda environment with all required packages (PyTorch, PyTorch Geometric, HuggingFace Transformers, W&B, and remaining pip dependencies); installing the package in editable mode (`pip install -e .`); authenticating with Weights & Biases; dataset preparation (all three datasets — FEMNIST, PubMed, SST2 — are downloaded automatically on first run, so no manual steps are required); and launching a hyperparameter sweep with `wandb sweep` and `wandb agent`. A configuration reference table lists every FedDecay-specific parameter alongside its type, default, and role. A Changes from Upstream section documents exactly which files were added or modified relative to the pfl\_bench fork and why, so a reader inspecting the diff knows where to look. The pfl\_bench README is linked for further FederatedScope documentation.

2. `federatedscope/core/trainers/trainer_decay.py` — Added a module-level docstring covering motivation, the hook-based implementation strategy, and the mathematical equivalence between the weight-level rescaling and a decayed learning rate. Every function now has a NumPy-style docstring. `_hook_decay_model` includes line-by-line inline comments walking through the weight-difference computation and both decay schemes, including why the step index starts at 0 so the first local update is always applied at full strength.

3. `federatedscope/core/configs/cfg_training.py` — The four config keys added for FedDecay (`trainer.model_on_batch_or_epoch`, `trainer.beta`, `trainer.finetune_beta`, `trainer.decay_scheme`) each now have an inline comment explaining their purpose, valid values, default behaviour, and relationship to the paper's experiments. With all defaults left unchanged, behaviour is identical to vanilla FedAvg.

4. `federatedscope/core/auxiliaries/trainer_builder.py` — The two `elif` branches that dispatch `federate.method = 'decay'` and `federate.method = 'FOMAML'` now each carry a comment identifying the corresponding paper method and pointing to the relevant trainer file.

---

### E3 — Black-and-white printing
**Feedback:** Do not refer to color in figure captions or text; all figures must be interpretable in black and white.

**Response:** The only color references found in any compiled `.tex` file were in the unused draft `content/figures/method-costs.tex` (`\textit{red}` / `\textit{blue}`), which has been updated to use positional descriptors (`\textit{top}` / `\textit{bottom}`), consistent with the caption used in the compiled document (`4c-costs.tex`). No color references remain in any compiled source.

---

### E4 — Figure font sizes
**Feedback:** All figures must have label and text font sizes of at least 10pt.

**Response:** This requires checking the source figures (`images/*.png`). Font sizes should be verified against the 10pt requirement before resubmission, and figures regenerated if necessary.

---

## Associate Editor

**Feedback:** Additional experiments in more extreme heterogeneity settings; whether coarse tuning of β is sufficient; discussion of expanding FedDecay to other FL algorithms.

**Response:** All three points are addressed through analysis of existing results and strengthened discussion. See the specific reviewer comments below.

- **Extreme heterogeneity** → Reviewer 1, Comment 1
- **Coarse β tuning** → Reviewer 2, Comment 3
- **Other FL algorithms** → Reviewer 1, Comment 3

---

## Reviewer 1

### R1.1 — Extreme heterogeneity scenarios
**Feedback:** The paper should examine more extreme heterogeneity scenarios or demonstrate the method's behavior in extreme corner cases.

**Response:** We argue that the existing experiments already span the relevant heterogeneity spectrum, and the theoretical framework covers all levels. No new experiments are needed.

The experiments include two qualitatively different heterogeneity regimes: FEMNIST uses natural author-level heterogeneity (moderate, with shared linguistic structure), while SST2 uses Dirichlet allocation with α = 0.4 — a level of synthetic, moderate-to-high non-IID data consistent with established benchmarks in the personalized federated learning literature (Chen et al., pFL-Bench, NeurIPS 2022; Zhao et al., 2018). Theorem 1 covers arbitrary heterogeneity Γ, so convergence guarantees hold at all non-IID levels.

The method's behavior in the extreme case is theoretically predicted, not merely conjectured. As inter-user similarity vanishes, the AvgGrad / AvgGradInner decomposition in Section 3 shows that there is no shared gradient signal to exploit, so the optimal β approaches 1, recovering FedAvg. Existing work already documents that FedAvg itself degrades substantially under extreme non-IID conditions (Zhao et al. 2018, cited as `fl-noniid`), and PFL methods become necessary in that regime — a well-established result that our framework aligns with rather than contradicts. Our method does not degrade below FedAvg: grid search returns β = 1 in this case.

**Changes made:**

1. `3c-decay.tex` (methodology, gradient decomposition summary) — Rewrote the passage on the extreme-heterogeneity regime to anchor the argument in the existing literature (`fl-noniid`), making the β → 1 recovery explicit and providing the β diagnostic for practitioners.

2. `5-conclusion.tex` (limitations) — Replaced self-limiting language ("our findings may not fully encapsulate") with a confident statement that the experiments cover the standard heterogeneity spectrum, that Theorem 1 covers arbitrary Γ, and that the method gracefully degrades to FedAvg in the extreme case. New citations are not introduced in the conclusion; the argument instead back-references Section 2 (related work) and Section 4 (experiments), where the supporting literature is already cited.

---

### R1.2 — Client-specific decay schedules
**Feedback:** Briefly investigate or comment on whether a client-specific decay schedule would be beneficial.

**Response:** Addressed in two locations, making the design choice feel deliberate rather than overlooked.

1. `3c-decay.tex` (methodology, after Equations 3–4) — A brief remark was added noting that a uniform global β is used for simplicity and theoretical tractability, and that client-specific schedules are a natural extension discussed in Section 5.

2. `5-conclusion.tex` (limitations) — A new paragraph explains the genuine challenges: per-client tuning of β_i requires either substantial additional communication overhead or adequate local validation data at each client, neither of which may be available in practice. Additionally, Theorem 1 assumes a uniform decay sequence; extending the convergence guarantees to heterogeneous per-client schedules requires non-trivial additional analysis. This is identified as a promising direction for future work.

Note: An earlier draft incorrectly cited "synchronization challenges in distributed settings" as a barrier to client-specific β. This has been corrected; the actual barriers are the hyperparameter tuning overhead and the convergence analysis extension described above.

---

### R1.3 — Extension to other FL algorithms (FedProx, FedAdam, FedYogi)
**Feedback:** A brief discussion of whether newer FL algorithms could leverage within-round learning rate decay would benefit readers.

**Response:** Addressed in two locations.

1. `2-background.tex` (related work) — The existing "several other works" passage was expanded to name SCAFFOLD, FedProx, FedAdam, and FedYogi explicitly and explain that each modifies either the global aggregation step or the local objective. Because FedDecay operates only on the within-round learning rate, these methods are structurally orthogonal and could in principle be combined with FedDecay for further gains.

2. `5-conclusion.tex` (limitations) — The final limitations paragraph now names FedProx, FedAdam, and FedYogi and identifies their combination with FedDecay as a natural avenue for future investigation.

**Citations upgraded:** The existing `fedprox`, `scaffold`, and `fed-opt` entries in `references.bib` were all arXiv preprints. These have been replaced with their proper published-venue citations:
- `fedprox` → MLSys 2020, *Proceedings of Machine Learning and Systems*, vol. 2, pp. 429–450 (duplicate arXiv entry also removed)
- `scaffold` → ICML 2020, *Proceedings of the 37th ICML*, PMLR vol. 119, pp. 5132–5143
- `fed-opt` → ICLR 2021, *International Conference on Learning Representations*

---

## Reviewer 2

### R2.1 — Communication rounds convergence
**Feedback:** The paper discusses computational efficiency but does not evaluate whether learning rate decay impacts convergence in terms of the number of communication rounds.

**Response:** Addressed in the costs section (`4c-costs.tex`). The text was revised to make the communication-rounds argument explicit: within-round decay does not alter the global aggregation step, so the number of communication rounds is unchanged. The O(N⁻¹) convergence rate established in Theorem 1 matches that of FedAvg. This is confirmed empirically: FedDecay terminates in the same number of training epochs as FedAvg on both FEMNIST and SST2.

---

### R2.2 — Heuristic for choosing FedDecay vs. PFL
**Feedback:** A heuristic or metric to guide users in choosing between FedDecay and more personalized alternatives would improve practical usability.

**Response:** A concrete heuristic is provided at two levels.

1. `3c-decay.tex` (methodology) — The β diagnostic is introduced at the point in the paper where the AvgGrad/AvgGradInner decomposition is established, giving it theoretical grounding: small β → similar users → FedDecay provides gains; β ≈ 1 → heterogeneous users → FedDecay recovers FedAvg and PFL should be considered.

2. `5-conclusion.tex` (limitations) — The practitioners' diagnostic is stated directly: use the cross-validated β as a proxy for data heterogeneity; values near 1 indicate PFL may be more appropriate.

---

### R2.3 — β sensitivity, coarse tuning sufficiency, and selection patterns
**Feedback:** Discuss sensitivity to β values; whether coarse tuning is generally sufficient; and patterns for selecting β based on data properties.

**Response:** A new concluding paragraph was added to Appendix A (`a-sensitivity.tex`), directly citing Table A.1:

- **Sensitivity:** Every β value not terminated early by hyperband outperforms FedAvg (β = 1) across all three datasets, showing the benefit is robust to the precise choice.
- **Coarse tuning sufficiency:** A practitioner using only {0.2, 0.4, 0.6} recovers a competitive solution on each dataset. Fine-grained search is not required.
- **Selection patterns:** Datasets with similar users (FEMNIST) favor smaller β; datasets with pronounced distributional heterogeneity (SST2) favor larger β. When prior knowledge about user similarity is available, it can guide the initial β choice and further reduce search cost.

---

## Additional Fixes

| File | Change |
|---|---|
| `3b-updates.tex` | Fixed typo: `ususer'shange` → `user's change` |
| `b-linear.tex` | Fixed figure caption: `Without-Round` → `Within-Round` |
| `figures/method-costs.tex` | Removed color descriptors (`\textit{red}`, `\textit{blue}`) → positional (`\textit{top}`, `\textit{bottom}`) |
| `references.bib` | `fedprox`: arXiv → MLSys 2020; `scaffold`: arXiv → ICML 2020; `fed-opt`: arXiv → ICLR 2021; duplicate `fedprox` entry removed |
