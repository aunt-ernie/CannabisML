#!/bin/bash

work=$1
stats=/mnt/Filbey/Ryan/currentProjects/SVM/Final/dmri/stats

${FSLDIR}/bin/fslmerge -t $stats/all_FA `$FSLDIR/bin/imglob $work/*/dwi/FA/dti_FA_to_FMRIB58.*`
cd $stats

# create mean FA
echo "creating valid mask and mean FA"
$FSLDIR/bin/fslmaths all_FA -max 0 -Tmin -bin mean_FA_mask -odt char
$FSLDIR/bin/fslmaths all_FA -mas mean_FA_mask all_FA
$FSLDIR/bin/fslmaths all_FA -Tmean mean_FA

# create skeleton
echo "skeletonising mean FA"
$FSLDIR/bin/tbss_skeleton -i mean_FA -o mean_FA_skeleton


echo "now view mean_FA_skeleton to check whether the default threshold of 0.2 needs changing, when running:"
echo "tbss_4_prestats <threshold>"
