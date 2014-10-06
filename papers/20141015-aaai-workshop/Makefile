%.tex: %.org
	emacsclient -n --eval "(progn (find-file \"$<\") (org-beamer-export-to-latex) (switch-to-buffer \"*terminal*\"))"

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

AuthorKit.zip:
	curl -O http://www.aaai.org/Publications/Templates/AuthorKit.zip
	unzip AuthorKit.zip

aaai.bst: AuthorKit.zip
	ln -s AuthorKit/aaai.bst .

aaai.sty: AuthorKit.zip
	ln -s AuthorKit/aaai.sty .
