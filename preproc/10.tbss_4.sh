#!/bin/bash

set -e

# Run for top level directory containing subject directories. Assumes ../stats exists.
WORK=$PWD

# Name of dti fit (or similar derivative) output. Omit extension or naming will be weird.
ALTIM=$1

# Base name of output directories.
OUT=$2

echo "upsampling alternative images into standard space"
# Apply FA warp to alternate image
for subj in $WORK/*; do
  if [[ ! -d $subj/dwi/$OUT ]]; then
    mkdir $subj/dwi/$OUT
  fi

  echo "applywarp -i $subj/dwi/${ALTIM} -o $subj/dwi/$OUT/${ALTIM}_to_FMRIB58 \
  -r $FSLDIR/data/standard/FMRIB58_FA_1mm -w $subj/dwi/FA/dti_FA_to_FMRIB58_warp"
done > ../10.tbss_4.txt

cat ../10.tbss_4.txt | parallel -j 30 {}

echo "merging all upsampled $ALTIM images into single 4D image"
${FSLDIR}/bin/fslmerge -t ../stats/all_$OUT \
`$FSLDIR/bin/imglob $WORK/*/dwi/$OUT/${ALTIM}_to_FMRIB58.*`
cd ../stats
$FSLDIR/bin/fslmaths all_$OUT -mas mean_FA_mask all_$OUT

echo "projecting all_$ALTIM onto mean FA skeleton"
thresh=`cat thresh.txt`
${FSLDIR}/bin/tbss_skeleton -i mean_FA -p $thresh mean_FA_skeleton_mask_dst \
${FSLDIR}/data/standard/LowerCingulum_1mm all_FA all_${OUT}_skeletonised -a all_$OUT

echo "now run stats - for example:"
echo "randomise -i all_${ALTIM}_skeletonised -o tbss_$ALTIM -m mean_FA_skeleton_mask -d design.mat -t design.con -n 500 --T2 -V"
echo "(after generating design.mat and design.con)"
