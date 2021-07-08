#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the predictors and CSA for all subjects of the ukbiobank project
#
# For usage, type: uk_get_subject_info -h
# Author: Sandrine Bédard

import os
import argparse
import numpy as np
import pandas as pd


# Dictionary of the predictors and field number correspondance
param_dict = {
        'eid': 'Subject',
        '31-0.0': 'Sex',
        '21003-2.0': 'Age',
        '12144-2.0': 'Height',
        '21002-2.0': 'Weight',
        '25000-2.0': 'Vscale',
        '25004-2.0': 'Volume ventricular CSF',
        '25006-2.0': 'Brain GM volume',
        '25008-2.0': 'Brain WM volume',
        '25009-2.0': 'Total brain volume norm',
        '25010-2.0': 'Total brain volume',
        '25011-2.0': 'Volume of thalamus (L)',
        '25012-2.0': 'Volume of thalamus (R)'
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects parameters from participant.tsv and CSA results from process_data.sh and writes them in data_ukbiobank.csv file in <path-output>/results",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-path-results',
                        required=True,
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the folder that will contain output files (processed data, results, log, QC).")
    parser.add_argument('-datafile',
                        required=False,
                        type=str,
                        default='participants.tsv',
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
    csa = pd.DataFrame(sc_data[['Filename', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32)
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def append_csa_to_df(df, csa, column_name):
    """
    Adds CSA values to dataframe for each subjects.
    Args:
        df (pandas.DataFrame): dataframe of parameters for all subjects with subjects' eid as row index.
        csa (pandas.DataFrame): dataframe of csa values with subjects' eid as row index.
        column_name (str): name of the new column to add.
    """
    # Loop through all subjects of Uk biobank
    for subject in df.index.tolist():
        # For subjects that have csa values,
        if subject in csa.index:
            # Set csa value for the subject
            df.loc[subject, column_name] = csa.loc[subject, 'MEAN(area)']


def compute_total_thalamus_volume(df):
    """
    Add a new column with the sum of right and left thalamus volume. Remove columns of right and left thalamus volume in df.
    Args:
        df (pandas.DataFrame): dataframe of parameters for all subjects.
    """
    df['Thalamus Volume'] = df['Volume of thalamus (L)'] + df['Volume of thalamus (R)']
    df.drop('Volume of thalamus (L)', axis='columns', inplace=True)
    df.drop('Volume of thalamus (R)', axis='columns', inplace=True)


def check_neuro_disease_history(df_disease, df):
    """
    Check if subject has history of nervous system disorders (fields https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=2406).
    Add 1 in column neuro_disease if it is the case.
    Args:
        df_disease (pandas.DataFrame):; dataframe with all fields.
        df (pandas.DataFrame): dataframe of parameters for all subjects with subjects' eid as row index.
    """
    fields = df_disease.columns.values.tolist()
    disease = [i for i in fields if i.startswith('13')]
    disease.append('eid')
    exclude = (df_disease[disease].set_index('eid')).dropna(0, how='all')
    for subject in df.index.tolist():
        # For subjects that have neuro disease,
        if subject in exclude.index.tolist():
            # Set 1 value for the subject
            df.loc[subject, 'neuro_disease'] = 1
        else:
            df.loc[subject, 'neuro_disease'] = 0
    df['neuro_disease'] = df['neuro_disease'].astype(np.int)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Open participant.tsv --> get data for subjects and selected predictors, create a dataframe.
    path_results = os.path.join(args.path_results, 'results')

    path_datafile = os.path.join(path_results, args.datafile)
    if os.path.splitext(path_datafile)[1] == '.tsv':
        raw_data = tsv2dataFrame(path_datafile)
    elif os.path.splitext(path_datafile)[1] == '.csv':
        raw_data = csv2dataFrame(path_datafile)

    # Initialize an empty dataframe with the predictors as columns
    df = pd.DataFrame(columns=param_dict.values())
    # Copy the raw data of the predictors into df
    for key, param in param_dict.items():
        df[param] = raw_data[key]

    # Initialize name of csv file of CSA in results folder
    path_csa_t1w = os.path.join(path_results, 'csa-SC_T1w.csv')

    # Set the index of the dataFrame to 'Subject'
    df = df.set_index('Subject')

    # Check if subject has history of neuro disease
    check_neuro_disease_history(raw_data, df)
    # Sum right and left thalamus volume
    compute_total_thalamus_volume(df)

    # Get csa values for T1w
    csa_t1w = get_csa(path_csa_t1w)

    # Add column to dataFrame of CSA values for T1w for each subject
    append_csa_to_df(df, csa_t1w, 'T1w_CSA')
    # Write a .csv file in <path_results/results> folder
    filename = 'data_ukbiobank.csv'
    df.to_csv(os.path.join(path_results, filename))


if __name__ == '__main__':
    main()
