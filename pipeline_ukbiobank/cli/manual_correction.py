#!/usr/bin/env python
#
# Script to perform manual correction of segmentations and vertebral labeling.
#
# For usage, type: python manual_correction.py -h
#
# Authors: Jan Valosek, Julien Cohen-Adad
# Adapted by Sandrine Bédard for cord CSA project UK Biobank

import argparse
import coloredlogs
import json
import os
import sys
import shutil
from textwrap import dedent
import time
import yaml

import utils

# Folder where to output manual labels, at the root of a BIDS dataset.
# TODO: make it an input argument (with default value)
FOLDER_DERIVATIVES = os.path.join('derivatives', 'labels')


def get_parser():
    """
    parser function
    """
    parser = argparse.ArgumentParser(
        description='Manual correction of spinal cord segmentation and vertebral labeling. '
                    'Manually corrected files are saved under derivatives/ folder (BIDS standard).',
        formatter_class=utils.SmartFormatter,
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-config',
        metavar="<file>",
        required=True,
        help=
        "R|Config yaml file listing images that require manual corrections for segmentation and vertebral "
        "labeling. 'FILES_SEG' lists images associated with spinal cord segmentation"
        "and 'FILES_LABEL' lists images associated with vertebral labeling. "
        "You can validate your yaml file at this website: http://www.yamllint.com/. If you only want to correct segmentation only, ommit 'FILES_LABEL' in the list. Below is an example yaml file:\n"
        + dedent(
            """
            FILES_SEG:
            - sub-1000032_T1w.nii.gz
            - sub-1000083_T2w.nii.gz
            FILES_LABEL:
            - sub-1000032_T1w.nii.gz
            - sub-1000710_T1w.nii.gz\n
            """)
    )
    parser.add_argument(
        '-path-in',
        metavar="<folder>",
        help='Path to the processed data. Example: ~/ukbiobank_results/data_processed',
        default='./'
    )
    parser.add_argument(
        '-path-out',
        metavar="<folder>",
        help="Path to the BIDS dataset where the corrected labels will be generated. Note: if the derivatives/ folder "
             "does not already exist, it will be created."
             "Example: ~/data-ukbiobank",
        default='./'
    )
    parser.add_argument(
        '-qc-only',
        help="Only output QC report based on the manually-corrected files already present in the derivatives folder. "
             "Skip the copy of the source files, and the opening of the manual correction pop-up windows.",
        action='store_true'
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Full verbose (for debugging)",
        action='store_true'
    )

    return parser

def get_function(task):
    if task == 'FILES_SEG':
        return 'sct_deepseg_sc'
    elif task == 'FILES_LABEL':
        return 'sct_label_utils'
    else:
        raise ValueError("This task is not recognized: {}".format(task))


def get_suffix(task, suffix=''):
    if task == 'FILES_SEG':
        return '_seg'+suffix
    elif task == 'FILES_LABEL':
        return '_labels'+suffix
    else:
        raise ValueError("This task is not recognized: {}".format(task))


def correct_segmentation(fname, fname_seg_out):
    """
    Copy fname_seg in fname_seg_out, then open ITK-SNAP with fname and fname_seg_out.
    :param fname:
    :param fname_seg:
    :param fname_seg_out:
    :param name_rater:
    :return:
    """
    # launch ITK-SNAP
    print("In ITK-SNAP, correct the segmentation, then save it with the same name (overwrite).")
    os.system('ITK-SNAP -g ' + fname + ' -s ' + fname_seg_out)


def correct_vertebral_labeling(fname, fname_label):
    """
    Open sct_label_utils to manually label vertebral levels.
    :param fname:
    :param fname_label:
    :param name_rater:
    :return:
    """
    message = "Click at the posterior tip of the disc between C2 and C3 vertebral levels, then click 'Save and Quit'."
    os.system('sct_label_utils -i {} -create-viewer 3 -o {} -msg "{}"'.format(fname, fname_label, message))


def create_json(fname_nifti, name_rater):
    """
    Create json sidecar with meta information
    :param fname_nifti: str: File name of the nifti image to associate with the json sidecar
    :param name_rater: str: Name of the expert rater
    :return:
    """
    metadata = {'Author': name_rater, 'Date': time.strftime('%Y-%m-%d %H:%M:%S')}
    fname_json = fname_nifti.rstrip('.nii').rstrip('.nii.gz') + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Logging level
    if args.verbose:
        coloredlogs.install(fmt='%(message)s', level='DEBUG')
    else:
        coloredlogs.install(fmt='%(message)s', level='INFO')

    # check if input yml file exists
    if os.path.isfile(args.config):
        fname_yml = args.config
    else:
        sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(args.config))

    # fetch input yml file as dict
    with open(fname_yml, 'r') as stream:
        try:
            dict_yml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # check for missing files before starting the whole process
    utils.check_files_exist(dict_yml, args.path_in)

    # check that output folder exists and has write permission
    path_out_deriv = utils.check_output_folder(args.path_out, FOLDER_DERIVATIVES)

    # Get name of expert rater (skip if -qc-only is true)
    if not args.qc_only:
        name_rater = input("Enter your name (Firstname Lastname). It will be used to generate a json sidecar with each "
                           "corrected file: ")

    # Build QC report folder name
    fname_qc = 'qc_corr_' + time.strftime('%Y%m%d%H%M%S')

    # TODO: address "none" issue if no file present under a key
    # Perform manual corrections
    for task, files in dict_yml.items():
        if files is not None:
            for file in files:
                # build file names
                subject = file.split('_')[0]
                contrast = 'anat'
                fname = os.path.join(args.path_in, subject, contrast, file)
                fname_label = os.path.join(
                    path_out_deriv, subject, contrast, utils.add_suffix(file, get_suffix(task, '-manual')))
                os.makedirs(os.path.join(path_out_deriv, subject, contrast), exist_ok=True)
                if not args.qc_only:
                    if os.path.isfile(fname_label):
                        # if corrected file already exists, asks user if they want to overwrite it
                        answer = None
                        while answer not in ("y", "n"):
                            answer = input("WARNING! The file {} already exists. "
                                           "Would you like to overwrite it? [y/n] ".format(fname_label))
                            if answer == "y":
                                do_labeling = True
                            elif answer == "n":
                                do_labeling = False
                            else:
                                print("Please answer with 'y' or 'n'")
                    else:
                        do_labeling = True
                    # Perform labeling for the specific task
                    if do_labeling:
                        if task in ['FILES_SEG']:
                            fname_seg = utils.add_suffix(fname, get_suffix(task))
                            shutil.copyfile(fname_seg, fname_label)
                            correct_segmentation(fname, fname_label)
                        elif task == 'FILES_LABEL':
                            if not utils.check_software_installed():
                                sys.exit("Some required software are not installed. Exit program.")
                            correct_vertebral_labeling(fname, fname_label)
                        else:
                            sys.exit('Task not recognized from yml file: {}'.format(task))
                        # create json sidecar with the name of the expert rater
                        create_json(fname_label, name_rater)

                # generate QC report (only for vertebral labeling or for qc only)
                if args.qc_only or task == 'FILES_LABEL':
                    os.system('sct_qc -i {} -s {} -p {} -qc {} -qc-subject {}'.format(
                        fname, fname_label, get_function(task) , fname_qc, subject))
                    # Archive QC folder
                    shutil.copy(fname_yml, fname_qc)
                    shutil.make_archive(fname_qc, 'zip', fname_qc)
                    print("Archive created:\n--> {}".format(fname_qc+'.zip'))

if __name__ == '__main__':
    main()
