PAPER_DIR = paper
TARGETS = jcgs tmlr jmlr

jcgs: $(PAPER_DIR)/2_main_jcgs.pdf
tmlr: $(PAPER_DIR)/1_main_tmlr.pdf
jmlr: $(PAPER_DIR)/0_main_jmlr.pdf

$(PAPER_DIR)/%.pdf: $(PAPER_DIR)/%.tex
	latexmk -pdf -interaction=nonstopmode -output-directory=$(PAPER_DIR) $<

clean:
	latexmk -C -output-directory=$(PAPER_DIR) $(PAPER_DIR)/*.tex

.PHONY: $(TARGETS) clean
