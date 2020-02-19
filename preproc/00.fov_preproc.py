#!/usr/bin/env python

import sys
import argparse
import numpy as np
import nibabel as nib


class reduceFOV:
    def __init__(self, anat, out, std):
        self.anat = anat
        self.out = out
        self.std = float(std)

    def reduce(self):
        img_data = nib.load(self.anat).get_data()

        # Remove padding along z-axis (if all zeros)
        zslice_means = []
        zslice_means.append(np.mean(img_data, (0, 1)))
        zslice_means = zslice_means[0]
        rm_slices = []
        for idx, slice in enumerate(zslice_means):
            if slice < 1:
                rm_slices.append(idx)
        shift = 0
        for zslice in rm_slices:
            img_data = np.delete(img_data, zslice-shift, axis=2)
            shift += 1

        # Remove padding along y-axis (if all zeroes)
        yslice_means = []
        yslice_means.append(np.mean(img_data, (0, 2)))
        yslice_means = yslice_means[0]
        rm_slices = []
        for idx, slice in enumerate(yslice_means):
            if slice < 1:
                rm_slices.append(idx)
        shift = 0
        for yslice in rm_slices:
            img_data = np.delete(img_data, yslice-shift, axis=1)
            shift += 1

        # Remove slice < 2 std dev from mean slice
        zslice_means = []
        zslice_means.append(np.mean(img_data, (0, 1)))
        zslice_means = zslice_means[0]
        mean_zslice = np.mean(zslice_means)
        sd_zslice = np.std(zslice_means)
        z_l_bound = mean_zslice - (self.std*sd_zslice)

        rm_slices = []
        for idx, slice in enumerate(zslice_means):
            if slice < z_l_bound:
                rm_slices.append(idx)
        shift = 0
        for zslice in rm_slices:
            img_data = np.delete(img_data, zslice-shift, axis=2)
            shift += 1

        yslice_means = []
        yslice_means.append(np.mean(img_data, (0, 2)))
        yslice_means = yslice_means[0]
        mean_yslice = np.mean(yslice_means)
        sd_yslice = np.std(yslice_means)
        y_l_bound = mean_yslice - (self.std*sd_yslice)
        rm_slices = []
        for yslice in rm_slices:
            img_data = np.delete(img_data, yslice-shift, axis=1)
            shift += 1
        for idx, slice in enumerate(yslice_means):
            if slice < y_l_bound:
                rm_slices.append(idx)
        shift = 0
        for yslice in rm_slices:
            img_data = np.delete(img_data, yslice-shift, axis=1)
            shift += 1

        # Save image
        new_image = nib.Nifti1Image(img_data, affine=np.eye(4))
        nib.save(new_image, self.out)


parser = argparse.ArgumentParser()
parser.add_argument("anat", help="Path to T1w original image.")
parser.add_argument("out", help="Path to file to write output to.")
parser.add_argument("std", help="Number of std dev away from slice mean to remove.")
args = parser.parse_args()

anat = args.anat
out = args.out
std = args.std
if anat.endswith('/'):
    anat = anat[:-1]
if out.endswith('/'):
    out = out[:-1]

reduceFOV(anat, out, std).reduce()
