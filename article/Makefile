PAPER_DIR = paper
TARGETS = jcgs

jcgs: $(PAPER_DIR)/main_jcgs.pdf

$(PAPER_DIR)/%.pdf: $(PAPER_DIR)/%.tex
	latexmk -pdf -bibtex -interaction=nonstopmode -output-directory=$(PAPER_DIR) $<

clean:
	latexmk -C -output-directory=$(PAPER_DIR) $(PAPER_DIR)/main_jcgs.tex

.PHONY: $(TARGETS) clean
