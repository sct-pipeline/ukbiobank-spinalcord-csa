#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the predictors and CSA for all subjects of the ukbiobank project
#
# For usage, type: get_subject_info -h
# Author: Sandrine Bédard

import os
import argparse
import csv
import json
import pandas as pd
import numpy as np
from datetime import date

# Dictionary of the predictors and field number correspondance
param_dict = {
        'eid':'Subject',
        '31-0.0':'Sex',
        '52-0.0':'Month of birth',
        '34-0.0':'Year of birth',
        '12144-2.0':'Height',
        '21002-2.0':'Weight',
        '25010-2.0':'Intracranial volume',
        '53-2.0': 'Date'
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects parameters from participant.tsv and CSA results from process_data.sh and writes them in data_ukbiobank.csv file in <path-output>/results",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-path-data',
                        required=True, 
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the folder that contains the data to be analyzed.")
    parser.add_argument('-path-output', 
                        required=True,
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the folder that will contain output files (processed data, results, log, QC).")
    parser.add_argument('-datafile',
                        required=False,
                        type=str,
                        default='participant.tsv', 
                        metavar='<filename>',
                        help="Name of the tsv file of the ukbiobank raw data. Default: participant.tsv")
    return parser


def tsv2dataFrame(filename):
    """
    Loads a .tsv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .tsv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .tsv file's data
    """
    data = pd.read_csv(filename, sep='\t')
    return data


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    data = pd.read_csv(filename)
    return data


def get_csa(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with CSA values sorted by subject eid.
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.Series): column of CSA values

    """
    sc_data = csv2dataFrame(csa_filename)
    csa = pd.DataFrame(sc_data[['Filename','MEAN(area)']]).rename(columns={'Filename':'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-37, -30).astype(int)
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def compute_age(df):
    """
    With the birth month and year of each subjects, computes age of the subjects at 2nd assessment
    Args:
        df (pd.dataFrame): dataframe of parameters for ukbiobank project
    Returns:
        df (pd.dataFrame): modified dataFrame with age
    """
    # Sperate year, month and day of 2nd assessment date
    df[['Year', 'Month', 'Day']] = (df['Date'].str.split('-', expand=True)).astype(int)
    # If the birth month is passed the 2nd assessment month, the age is this 2nd assessment year minus the birth year
    df.loc[df['Month of birth']<= df['Month'], 'Age'] = df['Year']- df['Year of birth']
    # If the birth month is not passed, the age is the 2nd assessment minus the birth year minus 1
    df.loc[df['Month of birth']> df['Month'], 'Age'] = df['Year'] - df['Year of birth'] -1
    # Delete the columns used to compute age
    df =  df.drop(columns = ['Year of birth', 'Month of birth', 'Year', 'Date', 'Month', 'Day'])
    # Format age as integer
    df['Age'] = df['Age'].astype(int)
    return df


def append_csa_to_df(df, csa, column_name):
    """
    Adds CSA values to dataframe for each subjects.
    Args:
        df (pandas.DataFrame): dataframe of parameters for all subjects with subjects' eid as row index
        csa (pandas.DataFrame): dataframe of csa values with subjects' eid as row index
        column_name (str): name of the new column to add
    """
    # Loop through all subjects of Uk biobank
    for subject in df.index.tolist():
        # For subjects that have csa values,
        if subject in csa.index:
            # Set csa value for the subject
            df.loc[subject, column_name] = csa.loc[subject, 'MEAN(area)']


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Open participant.tsv --> get data for subjects and selected predictors, create a dataframe.
    path_data  = os.path.join(args.path_data, args.datafile)
    raw_data = tsv2dataFrame(path_data)
 
    # Initialize an empty dataframe with the predictors as columns
    df = pd.DataFrame(columns = param_dict.values())
    # Copy the raw data of the predictors into df
    for key,param in param_dict.items():
        df[param] = raw_data[key]
    
    # Compute age and add an 'Age' column to df
    df = compute_age(df)

    # Initialize names of csv files of CSA in results file 
    path_results = os.path.join(args.path_output,'results')
    path_csa_t1w = os.path.join(path_results,'csa-SC_T1w.csv')
    path_csa_t2w = os.path.join(path_results,'csa-SC_T2w.csv')
    
    # Set the index of the dataFrame to 'Subject'
    df = df.set_index('Subject')

    # Get csa values for T1w and T2w
    csa_t1w = get_csa(path_csa_t1w)
    csa_t2w = get_csa(path_csa_t2w)

    # Add column to dataFrame of CSA values for T1w and T2w for each subject
    append_csa_to_df(df, csa_t1w, 'T1w_CSA')
    append_csa_to_df(df, csa_t2w, 'T2w_CSA')

    # Write a .csv file in <path_results/results> folder
    filename = 'data_ukbiobank.csv'
    df.to_csv(os.path.join(path_results,filename))

if __name__ == '__main__':
    main()
