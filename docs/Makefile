.PHONY: all

all: handy_svg

build_dir:
	mkdir -p build

handy_svg: build_dir
	latex -output-directory build tex/handy.tex
	dvisvgm --no-fonts build/handy.dvi -o svg/handy.svg
