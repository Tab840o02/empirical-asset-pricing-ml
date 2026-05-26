# Makefile for report/main.tex
# Requires: pdflatex and bibtex on PATH
# Usage: make          -> build PDF
#        make clean    -> remove auxiliary files
#        make distclean -> remove all generated files including PDF

MAIN    = main
PDFLATEX = pdflatex
BIBTEX  = bibtex
FLAGS   = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib tables/*.tex
	$(PDFLATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)

clean:
	$(RM) $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	       $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	       texput.log

distclean: clean
	$(RM) $(MAIN).pdf
