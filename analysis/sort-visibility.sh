#!/bin/sh

f=${1=/data/cubes/visibility-05ms-050mss-all.tsv}

sort -n -k7 <(sed '1d' $f) \
    | sed 's,\(r-hip\|r-shin\|l-shin\|l-hip\|r-knee\|l-knee\|r-heel\|l-heel\|t3\|t9\),\1\t,' \
    | sed 's|\([0-9]\{3\}\)[0-9]*|\1|g'
