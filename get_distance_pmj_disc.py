#!/usr/bin/env python
# -*- coding: utf-8
# Compute distance between PMJ and C2-C3 disc along centerline
#
# For usage, type: python get_distance_pmj_dics -h

# Authors: Sandrine Bédard

import argparse
import csv
from operator import sub
import numpy as np
import nibabel as nib
import os

NEAR_ZERO_THRESHOLD = 1e-6


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute distance between C2-C3 intervertebral disc and PMJ. Outputs .csv file with results. | Orientation needs to be RPI" )
    parser.add_argument('-centerline', required=True, type=str,
                        help="Input image.")
    parser.add_argument('-disclabel', required=True, type=str,
                        help="Ouput image.")
    parser.add_argument('-o', required=False, type=str,
                        default='pmj_disc_distance.csv',
                        help="Ouput csv filename.")

    return parser


def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected PMJ on centerline and cord centerline.
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param z_index: z index PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: distance from PMJ and corresponding indexes.
    """
    length = 0
    arr_length = [0]
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2][:z_index + 1]), axis=0)
    return arr_length


def main():
    parser = get_parser()
    args = parser.parse_args()

    disc_label = nib.load(args.disclabel)
    dim = disc_label.header['pixdim']
    px = dim[0]
    py = dim[1]
    pz = dim[2]
    # Create an array with centerline coordinates
    centerline = np.genfromtxt(args.centerline, delimiter=',')
    # Get C2-C3 disc coordinate
    c2_c3_disc = np.argwhere(disc_label.get_fdata() == 3)[0]
    # Compute distance from PMJ of the centerline
    arr_distance = get_distance_from_pmj(centerline, centerline[2].argmax(), px, py, pz)
    # Get the index of centerline array of c2c3 disc
    c2c3_index = np.abs(centerline[2] - c2_c3_disc[-1]).argmin()  # centerline doesn't necessarly start at the index 0 if the segmentation is incomplete
    distance_disc_pmj = arr_distance[:, c2c3_index][0]
    subject = os.path.basename(args.disclabel).split('_')[0]

    fname_out = args.o
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['Subject', 'C2C3_distance_PMJ']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [subject, distance_disc_pmj]
        spamwriter.writerow(line)


if __name__ == '__main__':
    main()
