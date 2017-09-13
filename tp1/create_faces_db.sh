#!/bin/bash

shopt -s nullglob
for filename in *.jpg *.png; do
	convert $filename -resize 92x112^ -gravity center -extent 92x112 ${filename:0:${#filename}-4}.pgm
done
