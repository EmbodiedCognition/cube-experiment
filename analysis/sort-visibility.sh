#!/bin/sh

file=${1:-/data/cubes/visibility-05ms-050mss-all.tsv}
col=${2:-7}

echo $col $file

head -1 $file | sed 's|marker|\tmarker\t|'
sed '1d' $file | sort -n -k$col \
    | sed 's,\(r-hip\|r-shin\|l-shin\|l-hip\|r-knee\|l-knee\|r-heel\|l-heel\|t3\|t9\),\1\t,' \
    | sed 's|\([0-9]\{3\}\)[0-9]*|\1|g'
