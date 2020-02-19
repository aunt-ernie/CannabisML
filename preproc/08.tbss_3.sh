#!/bin/bash

FA_dir=$1
g=dti_FA
f=FMRIB58
f_img=$FSLDIR/data/standard/FMRIB58_FA_1mm
o=${g}_to_$f

$FSLDIR/bin/applywarp -i $FA_dir/$g -o $FA_dir/$o -r $f_img -w $FA_dir/${g}_to_${f}_warp --rel
