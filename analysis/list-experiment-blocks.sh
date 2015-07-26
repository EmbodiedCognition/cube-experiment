#!/bin/bash

for subj in f9 68 ca d1 85 af 16 65 14 c9 3b
do
    echo -n "$subj block0["
    for block in 0 1 2 3 4 5
    do
        if ls /data/cubes/00-as-recorded-100Hz/*$subj/*-block0$block >/dev/null 2>&1
        then echo -n $block
        fi
    done
    echo "]"
done
