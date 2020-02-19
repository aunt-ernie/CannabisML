#!/bin/bash

FA_dir=$1
g=dti_FA
f=FMRIB58
f_img=$FSLDIR/data/standard/FMRIB58_FA_1mm
o=${g}_to_$f

if [[ -f $FA_dir/${o}_warp.msf ]]; then
  rm $FA_dir/${o}_warp.msf
  touch $FA_dir/${o}_warp.msf
else
  touch $FA_dir/${o}_warp.msf
fi

$FSLDIR/bin/fsl_reg $FA_dir/$g $f_img $FA_dir/${g}_to_$f -e -FA
