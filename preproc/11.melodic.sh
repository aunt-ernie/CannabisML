#!/bin/bash

nohup melodic -i all_FA_skeletonised.nii.gz -o dim_28/all_FA_skeletonised_ICA \
-m mean_FA_skeleton_mask.nii.gz -d 28 -v --nobet --report --guireport=dim_28/report_FA.html \
--mmthresh=0.5 > ICA_FA.out 2> dim_28/ICA_FA.err &

nohup melodic -i all_MD_skeletonised_1000.nii.gz -o dim_28/all_MD_skeletonised_ICA \
-m mean_FA_skeleton_mask.nii.gz -d 28 -v --nobet --report --guireport=dim_28/report_MD.html \
--mmthresh=0.5 > dim_28/ICA_MD.out 2> dim_28/ICA_MD.err

nohup melodic -i all_RD_skeletonised_1000.nii.gz -o dim_28/all_RD_skeletonised_ICA \
-m mean_FA_skeleton_mask.nii.gz -d 28 -v --nobet --report --guireport=dim_28/report_RD.html \
--mmthresh=0.5 > dim_28/ICA_RD.out 2> dim_28/ICA_RD.err &

nohup melodic -i all_AD_skeletonised_1000.nii.gz -o dim_28/all_AD_skeletonised_ICA \
-m mean_FA_skeleton_mask.nii.gz -d 28 -v --nobet --report --guireport=dim_28/report_AD.html \
--mmthresh=0.5 > dim_28/ICA_AD.out 2> dim_28/ICA_AD.err

