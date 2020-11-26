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

# Dictionary of the predictors and field number correspondance| TODO: add to participants.tsv
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
        description="Gets the subjects parameters and CSA results from process_data.sh and writes them in data_ukbiobank.csv file in <path-output>/results",
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
                        default='subjects_gbm3100.csv',
                        metavar='<filename>',
                        help="Name of the csv file of the ukbiobank raw data. Default: subjects_gbm3100.csv")
    return parser

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
    returns a panda dataFrame with the subjects' eid and CSA values
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.dataFrame): dataframe of CSA values

    """
    sc_data = csv2dataFrame(csa_filename)
    csa = sc_data['MEAN(area)']
    return csa

def compute_age(df):
    """
    With the birth month and year of each subjects, computes age of the subjects at 2nd assessment
    Args:
        df (pd.dataFrame): dataframe of parameters for ukbiobank project
    Returns:
        df (pd.dataFrame): modified dataFrame with age
    """
    # Sperates year, month and day of 2nd assessment date
    df[['Year', 'Month', 'Day']] = (df['Date'].str.split('-', expand=True)).astype(int)
    # If the birth month is passed the 2nd assessment month, the age is this 2nd assessment year minus the birth year
    df.loc[df['Month of birth']<= df['Month'], 'Age'] = df['Year']- df['Year of birth']
    # If the birth month is not passed, the age is the 2nd assessment minus the birth year minus 1
    df.loc[df['Month of birth']> df['Month'], 'Age'] = df['Year'] - df['Year of birth'] -1
    # Deletes the columns used to compute age
    df =  df.drop(columns = ['Year of birth', 'Month of birth', 'Year', 'Date', 'Month', 'Day'])
    # Formats age as integer
    df['Age'] = df['Age'].astype(int)
    return df

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Open <datafile>.csv --> gets data for subjects and selected predictors, creates a dataframe.
    path_data  = args.path_data + '/' +args.datafile
    raw_data = csv2dataFrame(path_data)
    # Initialize an empty dataframe with the predictors as columns
    df = pd.DataFrame(columns = param_dict.values())
    # Copies the raw data of the predictors into df
    for key,param in param_dict.items():
        df[param] = raw_data[key]
    
    #Computes age and adds an 'Age' column to df
    df = compute_age(df)
    # Initializes names of csv files of CSA in results file --> maybe there is an other way 
    path_results = args.path_output+'/results/'
    t1_csaPath = path_results+'csa-SC_T1w.csv'
    t2_csaPath = path_results+'csa-SC_T2w.csv'
    
    #Gets data frame of CSA for T1w and T2w
    df['T1w_CSA'] = get_csa(t1_csaPath)
    df['T2w_CSA'] = get_csa(t2_csaPath)

    #Sets the index of the dataFrame to 'Subject'
    df = df.set_index('Subject')

    # Writes a .csv file in <path_results/results> folder
    filename = 'data_ukbiobank.csv'
    df.to_csv(path_results+filename)

if __name__ == '__main__':
    main()
