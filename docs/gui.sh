#!/bin/bash

i=0
for file in Screenshot*.png*; do
    if [[ "$i" == 0 ]]; then
        mv "$file" $1-gui-crystal.png;
        echo mv "$file" $1-gui-crystal.png;
    elif [[ "$i" == 1 ]]; then
        mv "$file" $1-gui-intensity.png;
        echo mv "$file" $1-gui-intensity.png;
    elif [[ "$i" == 2 ]]; then
        mv "$file" $1-gui-refinement.png;
        echo mv "$file" $1-gui-refinement.png;
    elif [[ "$i" == 3 ]]; then
        mv "$file" $1-gui-correlations.png;
        echo mv "$file" $1-gui-correlations.png;
    elif [[ "$i" == 4 ]]; then
        mv "$file" $1-gui-recalculation.png;
        echo mv "$file" $1-gui-recalculation.png;
    fi
    i=$((i+1))
done
