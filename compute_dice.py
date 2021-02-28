#!/usr/bin/env python
# -*- coding: utf-8
# Computes mean dice coefficient accross manual segmentations from candidates and ground truth segmentation. 
#
# For usage, type: python compute_dice.py -h
#
# Authors: Sandrine Bédard

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import shutil
import pipeline_ukbiobank.utils as utils
from textwrap import dedent

FNAME_LOG = 'log_dice_coeff.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes dice coefficient between manual segmentations of candidates and ground truth segmentations.",
        prog=os.path.basename(__file__).strip('.py'),
        formatter_class=SmartFormatter
        )
    parser.add_argument('-path-ref',
                        required=True, 
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the derivative folder of the ground truth segmentations. Example: derivatives/")
    parser.add_argument('-path-seg', 
                        required=True,
                        type=str,
                        metavar='<dir_path>',
                        help=
                        "R|Path to the folder including all manual segmentations from candidates.\n"
                        "Example of structure of the folder:\n"
                        + dedent(
                        """
                        candidates_segmentations
                        ├── surname_name1
                        |    └── derivatives
                        ├── surname_name2
                        |    └── derivatives                        
                        ...
                        """
                        ))
    parser.add_argument('-path-out',
                        required=False,
                        type=str,
                        default='./', 
                        metavar='<filename>',
                        help="Path where results will be written.")
    return parser


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def compute_dice(fname_ref_seg, fname_manual_seg):
    """
    Computes dice coefficient between the ground truth segmentation and a candidate's manual segmentation.
    Args:
        fname_ref_seg (str): file name of the ground truth segmentation.
        fname_manual_seg (str): file name of the segmentation from a candidate.
    Returns:
        dice (float): dice coefficient.
    """
    # Get the name and extension of the reference segmentation.
    stem, ext = splitext(fname_ref_seg)
    # Creates a temporary copy of the segmentation. Note: sct_dice_coefficient can't compute dice coefficient of files with the same name.
    ref_copy = os.path.join(stem + '-tmp'+ ext)
    shutil.copyfile(fname_ref_seg, ref_copy)  # Creates a copy of the ref seg.

    # Compute dice coefficient
    os.system('sct_dice_coefficient -i ' + fname_manual_seg + ' -d ' +  ref_copy + ' -o dice_coeff.txt')
    os.remove(ref_copy) # Remove copy of ref seg
    # Read the .txt file with the dice coeff
    with open('dice_coeff.txt', 'r') as reader:
        text = reader.read()
        dice = float(text.split()[-1])
    os.remove('dice_coeff.txt')  # Delete .txt file
    return dice


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Dump log file there
    path_log = os.path.join(args.path_out, FNAME_LOG)
    if os.path.exists(path_log):
        os.remove(path_log)
    fh = logging.FileHandler(path_log)
    logging.root.addHandler(fh)

    # Check if SCT is installed
    if not utils.check_software_installed():
        sys.exit("SCT is not installed. Exit program.")
    
    # Initialize empty DataFrame
    df = pd.DataFrame()

    # Loop through candidates
    for candidate in os.listdir(args.path_seg):
        path_manual_seg = os.path.join(args.path_seg, candidate, 'derivatives', 'labels')
        # Loop through subjects
        for subject in os.listdir(path_manual_seg):
            # Loop through files in anat/ folder
            for filename in os.listdir(os.path.join(path_manual_seg, subject, 'anat')):
                if filename.endswith('.nii.gz'):  # Is there another type to include?
                    # Get path of manual segmentation
                    manual_seg = os.path.join(path_manual_seg, subject, 'anat', filename)
                    # Get path of reference segmentation
                    ref_seg = os.path.join(args.path_ref,'labels', subject, 'anat', filename)
                    # Compute dice coefficient
                    dice = compute_dice(ref_seg, manual_seg)
                    # Add a row to the DataFrame with dice coefficient
                    df.loc[filename, candidate] = dice
    
    # Compute mean dice coefficient for all segmentations from a candidate
    df.loc['mean dice coeff',:] = df.mean(axis=0)

    # Write dataframe to log
    logger.info('Dice coefficients are:\n{}'.format(df))

if __name__ == '__main__':
    main()
