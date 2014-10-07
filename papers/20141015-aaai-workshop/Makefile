%.tex: %.org
	emacsclient -n --eval "(progn (find-file \"$<\") (org-latex-export-to-latex) (switch-to-buffer \"*terminal*\"))"

%.pdf: %.tex aaai.bst aaai.sty
	latexmk -pdf $<

.PHONY: all clean

all: paper.pdf

clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.nav
	rm -f *.out
	rm -f *.pdf
	rm -f *.snm
	rm -f *.toc
	rm -f *.vrb
	rm -f AuthorKit.zip
	rm -f aaai.bst
	rm -f aaai.sty

AuthorKit.zip:
	curl -O http://www.aaai.org/Publications/Templates/$@

aaai.%: AuthorKit.zip
	unzip -p $< AuthorKit/$@ > $@
