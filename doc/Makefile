# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pysal
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples/

dash:
	rm -rf PySAL.docset
	doc2dash --name=PySAL  --icon=_static/images/pysal_favicon.png _build/html 
	rm -rf ~/.local/share/Zeal/Zeal/docsets/PySAL.docset
	mkdir ~/.local/share/Zeal/Zeal/docsets/PySAL.docset
	cp -R PySAL.docset/ ~/.local/share/Zeal/Zeal/docsets/

