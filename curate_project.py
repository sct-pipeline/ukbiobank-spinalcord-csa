#!/usr/bin/env python
# -*- coding: utf-8
# Function to parse UK Biobank data
#
# Author: Alexandru Foias


import os
import argparse
import shutil

def get_parser():
    parser = argparse.ArgumentParser(
        description="Curate MRI data from UKBioBank" )
    parser.add_argument('-path-in', required=True, type=str,
                        help="Path to input UKBioBank dataset, which contains all the subject folders.")
    parser.add_argument('-path-out', required=True, type=str,
                        help="Path to output BIDS dataset, which contains all the 'sub-' folders.")

    return parser

def unzip_helper (path_in,path_out, subject):
    # Contrast dictionary of contrasts and desired files
    contrast_dictionary = {'T1': ['_20252_2_0.zip','T1/T1.nii.gz','T1/T1.json'],
                           'T2': ['_20253_2_0.zip','T2_FLAIR/T2_FLAIR.nii.gz','T2_FLAIR/T2_FLAIR.json']
                           }

    bids_filename_dictionary = {'T1.nii.gz': 'T1w.nii.gz',
                                'T1.json': 'T1w.json',
                                'T2_FLAIR.nii.gz': 'T2w.nii.gz',
                                'T2_FLAIR.json': 'T2w.json'
                                }
    path_output_folder_subject = os.path.join(path_out, 'sub-' + subject)

    # Loop across contrasts
    for contrast in contrast_dictionary.keys():
        path_archive = os.path.join(path_in, subject, 'zip', subject + contrast_dictionary[contrast][0])
        # Loop across different desired files for selected contrast
        for idx in range(1,len(contrast_dictionary[contrast])):
            path_file_in = contrast_dictionary[contrast][idx]
            command_image = 'unzip -j ' + path_archive + ' ' + path_file_in + ' -d ' + path_output_folder_subject
            os.system(command_image)
            filename_interm_in = path_file_in.split('/')[1]
            path_file_interm_in = os.path.join(path_output_folder_subject, filename_interm_in)

            #Create BIDS structure
            if not os.path.exists(os.path.join(path_output_folder_subject, 'anat')):
                os.makedirs(os.path.join(path_output_folder_subject, 'anat'))
            path_file_bids_out = os.path.join(path_output_folder_subject, 'anat', 'sub-' + subject + '_' + bids_filename_dictionary[filename_interm_in])
            os.system('mv ' + path_file_interm_in + ' ' + path_file_bids_out)

def main():
    # Parse input arguments
    parser = get_parser()
    args = parser.parse_args()
    # if os.path.exists(args.path_out):
    #     shutil.rmtree (args.path_out)
    # os.makedirs (args.path_out)
    list_subjects = [x for x in os.listdir(args.path_in) if os.path.isdir(os.path.join(args.path_in,x))]
    for subject in list_subjects:
        unzip_helper(args.path_in, args.path_out, subject)

if __name__ == '__main__':
    main()
