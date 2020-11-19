#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the predictors and CSA for the subject for th ukbiobank project
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
import pipeline_ukbiobank.cli.select_subjects as select_subjects

# Dictionary of the parameters and field number correspondance
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

csa_dict = {
    'MEAN(area)':('T1w_CSA','T2w_CSA')
}
def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects info and writes it in data_ukbiobank.csv file",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-datafile', required=True, type=str,
                        help="Name of the csv file of the ukbiobank data.")
    parser.add_argument('-path-data-output', required=True, type=str,
                        help="Name of the output file of the results of process_data.sh.")
    return parser


def get_csa(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with the subjects' eid and CSA values
    Args
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.dataFrame): dataframe of subjects eid ans CSA value

    """
    sc_data = select_subjects.load_participant_data_file(csa_filename)
    sc_data['Filename'] = sc_data['Filename'].str[-28:-21]
    csa= sc_data[['Filename', list(csa_dict.keys())[0]]]
    return csa

def compute_age(df):
    """
    With the birth month and year of each subjects, computes age of patient at 2nd assesment
    Args
        df (pd.dataFrame): dataframe of parameters for ukbiobank project
    """
    # Sperates year, month and day of 2nd assesment 
    df[['Year', 'Month', 'Day']] = (df['Date'].str.split('-', expand=True)).astype(int)
    # If the birth month is passed today's month, the age is this year minus the birth year
    df.loc[df['Month of birth']<= df['Month'], 'Age'] = df['Year']- df['Year of birth']
    # If the birth month is'nt passed , the age is this year minus the birth year minus 1
    df.loc[df['Month of birth']> df['Month'], 'Age'] = df['Year'] - df['Year of birth'] -1
    # Deletes the colums used to compute age
    df =  df.drop(columns = ['Year of birth', 'Month of birth', 'Year', 'Date', 'Month', 'Day'])
    # Formats age as int
    df['Age'] = df['Age'].astype(int)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Open <datafile>.csv --> get data for subjects and selected parameters Creates a dataframe.
    raw_data = select_subjects.load_participant_data_file(args.datafile)
    # Initialize an empty dataframe with the parameters as columns
    df = pd.DataFrame(columns = param_dict.values())
    # Copies the raw data of the parameters into df
    for key,param in param_dict.items():
        df[param] = raw_data[key]
    
    #Computes age and adds an 'Age' column to df
    compute_age(df)

    # Initiates name of csv files in results --> maybe there is an other way 
    path_results = args.path_data_output+'/results/'
    t1_csaPath = path_results+'csa-SC_T1w.csv'
    t2_csaPath = path_results+'csa-SC_T2w.csv'
    
    #Get data frame of subject eid and csa for T1w and T2w
    t1_csa =(get_csa(t1_csaPath)).set_index('Filename')
    t2_csa = (get_csa(t2_csaPath)).set_index('Filename')

    contrasts = list(csa_dict.values())[0]
    #Change the index to subject eid
    df = df.set_index('Subject')
    # For all subjects in csa file, puts de CSA value in df for T1w_CSA and T2W_CSA
    for sub in t1_csa.index:
        df.loc[int(sub), contrasts[0]] = t1_csa.at[sub,'MEAN(area)']
        df.loc[int(sub), contrasts[1]] = t2_csa.at[sub,'MEAN(area)']
    
    # Writes a .csv file in /results folder
    filename = 'data_ukbiobank.csv'
    df.to_csv(path_results+filename)

if __name__ == '__main__':
    main()
