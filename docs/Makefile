.PHONY: all

all: handy_svg

build_dir:
	mkdir -p build

handy_svg: build_dir
	latex -output-directory build tex/ball_dynamics.tex
	dvisvgm --no-fonts build/ball_dynamics.dvi -o svg/ball_dynamics.svg
