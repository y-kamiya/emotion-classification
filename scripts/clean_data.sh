#!/bin/bash

input_tsv=$1

tmp_dir=$(mktemp -d)
trap 'rm -rf $temp_dir' EXIT

cat $input_tsv | while read line; do echo -n $line | wc -c; done > $tmp_dir/bytes
cat $input_tsv | while read line; do echo -n $line | wc -m; done > $tmp_dir/multibytes

paste $tmp_dir/bytes $tmp_dir/multibytes $input_tsv \
| awk -F'\t' '$2*1.5 < $1 { print $3"\t"$4 }'
