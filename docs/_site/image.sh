#!/bin/bash

for file in website-*; do
    #rename -f 's/website-//' "$file"
    mv ${file} `echo ${file} | sed 's/website-//'`
done

for file in *.pdf; do
    if [[ "$file" -nt "${file%.pdf}.svg" ]]; then
        pdf2svg "$file" "${file%.pdf}.svg"
    fi
done
