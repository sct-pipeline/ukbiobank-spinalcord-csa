#!/usr/bin/env python
# -*- coding: utf-8
# Compares multiple yml list with a reference list.
#
# For usage, type: python compare_yml_list.py -h
#
# Authors: Sandrine Bédard

import argparse
import logging
import os
import sys
import yaml
import pandas as pd
from textwrap import dedent

FNAME_LOG = 'log_yml_comparasion.txt'

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
        description="Compares multiple .yml lists of subjects to perform manual segmentation with a reference list and outputs the results",
        prog=os.path.basename(__file__).strip('.py'),
        formatter_class=SmartFormatter
        )
    parser.add_argument('-ref-list',
                        required=True,
                        type=str,
                        metavar='<file>',
                        help=
                        "R|Filename of the reference .yml list.\n"
                        "The .yml file should have this structure:\n"
                        + dedent(
                        """
                        FILES_SEG:
                        - sub-1000032_T1w.nii.gz
                        - sub-1000083_T2w.nii.gz
                        """))
    parser.add_argument('-path-lists',
                        required=True,
                        type=str,
                        metavar='<dir_path>',
                        help="Folder containing .yml lists to compare")
    parser.add_argument('-path-out',
                        required=False,
                        default='./',
                        metavar='<dir_path>',
                        help='Path where results will be written.')
    return parser
      

def read_yml(filename):
    """
    Reads a .yml file and returns a dict of its content or 0 if an error occured while loading the file.
    Args:
        filename (str): Name of the .yml file to read.
    Returns:
        dict_yml (dict): Dictionnary of the .yml file.
            or
        format_error (bool): 0 if an error occured.
    """
    with open(filename, 'r') as stream:
        try:
            dict_yml = yaml.safe_load(stream)
            return dict_yml
        except yaml.YAMLError as exc:
           format_error = 0 
           logger.info(exc) 
           return format_error


def check_FILESEG(ref_dict, list_dict): # Maybe change name source
    """
    Checks if "FILESEG:" is at the beginning of the .yml file.
    Args:
        ref (dict): dictionnary of the reference .yml file.
        source (dict): dictionnary of the .yml file to compare.
    Returns:
        has_fileseg (bool): True if "FILESEG" is at the begining of the .yml file of source.
    """
    # Check the type of list_dict. (If 'FILESEG' was ommited, will be a list not a dict)
    if  isinstance(list_dict, dict):
        # Check if key in list_dict is the same as in the reference list. (FILESEG)
        if list_dict.keys() == ref_dict.keys():
            has_fileseg = True
            logger.info('FILESEG is the first line of the file.')
        else:
            has_fileseg = False
            logger.info('FILESEG wrongly nameed {}.'.format(list(list_dict.keys())[0]))
    else:
        has_fileseg = False
        logger.info('Missing FILESEG.')
    return has_fileseg
   

def compare_lists(ref_dict, list_dict):
    """
    Compares two dictionnary and retruns the number of file identified and correctly identified.
    Args:
        ref_dict (dict): Dictionnary of the reference .yml file.
        list_dict (dict): Dictionnary of the .yml list to compare. Note: could be a list if 'FILESEG' is missing.
    Returns:
        n_files_identified (str): Ratio of identified files to correct by number of files that their should be
        n_right_files (str): Ratio of files from those identified that truly need manual segmentation.
    """
    # Check if list_dict is a dict and not a list (if missing FILESEG).
    ref_list = list(ref_dict.values())[0]
    if  isinstance(list_dict, dict):
        list_yml = list(list_dict.values())[0]
    else:
        list_yml = list_dict
    
    total_selected = len(list_yml) # Compute number of files identified that need manual segmentation
    total_ref = len(ref_list) # Compute number of files that truly need manual segmentation
    # Initialization
    n_true = 0 # Number of correctly identified files
    right_files = [] # List of files that are identified correctly

    # Loop through both list to compare
    for filename in list_yml:
        for true_filename in ref_list:
            if filename == true_filename:
                n_true = n_true + 1
                right_files.append(filename)
    
    # Get filnames wrongly identified
    wrong_files = list_yml.copy()
    for filename in right_files:
        wrong_files.remove(filename)

    n_files_identified = '{}/{}'.format(total_selected, total_ref) # Ratio of idenified files
    n_right_files = '{}/{}'.format(n_true, total_selected) # Ratio of correctly identified files
    logger.info('- Number of files identified = {}'.format(n_files_identified))
    logger.info('- Number of right files = {}'.format(n_right_files))
    logger.info('- Files wrongly identified are: {}'.format(wrong_files))
    return n_files_identified, n_right_files


def df_to_csv(df, filename):
    """
    Saves a Dataframe as a .csv file.
    Args:
        df (panda.DataFrame)
        filename (str): Name of the output .csv file.
    """
    df.to_csv(filename)
    logger.info('Created: ' + filename)


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

    # Read ref yml file
    ref_dict = read_yml(args.ref_list)

    # Initialize lists for results
    names = [] # List of all filenames
    formating = [] # Results of loading .yml files
    fileseg = [] # Results: if FILESEG is at the beginnig of the .yml file
    nb_files = [] # Number of files identified for manual correction
    nb_true = [] # Number of files from those identified that truly need manual segmentation

    # Loop through all yml lists to compare with ref.
    for filename in os.listdir(args.path_lists):
        # Get name of the filename for identification
        name  = filename[:-4]
        names.append(name) # Adds name to list
        logger.info('\nComparing {} ...'.format(filename))
        
        # Read yml file.
        list_dict = read_yml(os.path.join(args.path_lists,filename))

        # If yml file was not properly loaded, doesn't perform comparasion
        if list_dict == 0 : # There was an error when loading .yml file
            formating.append('ERROR')
            fileseg.append('-')
            nb_files.append('-')
            nb_true.append('-')
            logger.info('{} has the wrong format. Unable to compare files.'.format(filename))
        else:
            # Perform comparasion
            formating.append('OK') # Format of .yml file is ok, can continue
            # Check if 'FILESEG' is at the beginning of the file
            is_FILESEG = check_FILESEG(ref_dict, list_dict)
            fileseg.append(is_FILESEG)
            # Compares lists
            n_total, n_right = compare_lists(ref_dict, list_dict)
            nb_files.append(n_total)
            nb_true.append(n_right)


    columns = ['FORMAT', 'FILESEG', 'Nb_files_identified', 'Nb_right_files']
    df = pd.DataFrame(index=names, columns= columns)
    df['FORMAT'] = formating
    df['FILESEG'] = fileseg
    df['Nb_files_identified'] = nb_files
    df['Nb_right_files'] = nb_true
    # Save results to .csv file
    path_results = os.path.join(args.path_out, 'results.csv')
    df_to_csv(df, path_results)
    logger.info('Comparison completed.')

if __name__ == '__main__':
    main()