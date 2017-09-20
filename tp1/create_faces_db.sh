#!/bin/bash

shopt -s nullglob
let a=1
for filename in *.jpg *.png *.JPG; do
	convert $filename -resize 92x112^ -gravity center -extent 92x112 $a.pgm
	a=$((a+1))
done
