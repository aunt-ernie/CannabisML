#!/bin/bash

cd /mnt/Filbey/Ryan/currentProjects/SVM/Final/dmri/work

for i in *; do
  cd $i/dwi
  tbss_1_preproc dti_FA.nii.gz
  cd ../../
done
