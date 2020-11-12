#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the selected subject list .yml
#
# For usage, type: select_subjects -h

import os
import argparse
import pipeline_ukbiobank
import csv
import yaml
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(
        description="Genrates a selected subjects list (.yml). A subject is selected if it has data for all the chosen parameters",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-subjects_datafile', required=True, type=str,
                        help="Name of the .csv file containing all the subjects feilds data.")
    parser.add_argument('-parameters', required=True, type=str,
                        help="Name of the .txt file containing all the selelected parameters' feild id.")                    
    return parser

def load_participant_data_file(filename):
    """
    Load the participaant datafile and build pandas DF of participants
    :return:data: pandas dataframe
    """
    data = pd.read_csv(filename)
    return data

def main ():
    parser = get_parser()
    args = parser.parse_args()

    #Load the .txt file containing the feild ID of each chosen parameter
    with open(args.parameters, 'r') as f:
        parameters = f.read().splitlines()
    #Load the .csv file containing all the data of the subjects
    df = load_participant_data_file(args.subjects_datafile)
    #Select the data for the chosen parameters
    df_short = df[parameters]
    #Removes a subject if it has an empty feild among the parameters
    df_updated = df_short.dropna(0,how = 'any').reset_index(drop=True)
    #Generates a list of the remaining subjects' eid 
    select_subjects = df_updated[parameters[0]].to_list()
    #Writes a .yml file with the selected subjects
    with open('selected_subjects.yml', 'w') as f:
        yaml.dump(select_subjects, f)

if __name__ == '__main__':
    main()