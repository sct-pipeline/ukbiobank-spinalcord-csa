#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the user info from ??
#
# For usage, type: get_subject_info -h
#À voir si utile, peut-être juste aller chercher les données dans csa.csv quand tout est traité

import os
import argparse
import csv
import json
import pandas as pd
import numpy as np
from datetime import date
import pipeline_ukbiobank.cli.select_subjects as select_subjects

param_dict = {
        'eid':'Subject',
        '31-0.0':'Sex',
        '52-0.0':'Month of birth',
        '34-0.0':'Year of birth',
        '12144-2.0':'Height',
        '21002-2.0':'Weight',
        '25010-2.0':'Intracranial volume',
         
    }
csa_dict = {
    'MEAN(area)':('T1w_CSA','T2w_CSA')
}
def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects info and writes it in data.csv file",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-parameters', required=True, type=str,
                        help="Name of the .txt file that contain the parameters.")
    parser.add_argument('-datafile', required=True, type=str,
                        help="Name of the csv file of the data.")
    parser.add_argument('-path-data-output', required=True, type=str,
                        help="Name output file of images.")
    return parser


def get_csa(csa_filename):
    sc_data = select_subjects.load_participant_data_file(csa_filename)
    #csa = sc_data[csa_dict.keys()]
    sc_data['Filename'] = sc_data['Filename'].str[-28:-21]
    csa= sc_data[['Filename', list(csa_dict.keys())[0]]]
    return csa

def main():
    parser = get_parser()
    args = parser.parse_args()
    current = os.getcwd()

    #1 .Create data_ukbiobank dataframe with the parameters as the columns
    #Lire feild id instead
    
    

    #2 . with participant.csv and exclude, get final list (peut-être comme dernière étape?)
    #3. open subjects_gbm3100.csv --> get data for subjects and selected parameters
    #Load the .txt file containing the feild ID of each chosen parameter (make a function for both select_subjects and get subject info)
    
    with open(args.parameters, 'r') as f: #pas sur si c'est nécéssaire
        parameters = f.read().splitlines()

    raw_data = select_subjects.load_participant_data_file(args.datafile)
    df = pd.DataFrame(columns = param_dict.values())
    
    #Adding subject number
    df = pd.DataFrame(columns = param_dict.values())
    for key,param in param_dict.items():
        df[param] = raw_data[key]
    
    #Comuptutre age

    #get date
    #Faire un fonction!!
    today = date.today()
    df.loc[df['Month of birth']<= today.month, 'Age'] = today.year - df['Year of birth']
    df.loc[df['Month of birth']> today.month, 'Age'] = today.year - df['Year of birth'] -1
    df['Age'] = df['Age'].astype(int)

    #4 read t1csa.csv and t2wcsa.csv --> get cord csa, write in data_ukbiobank.csv
    os.chdir(args.path_data_output+'/results')

    t1_csaPath = 'csa-SC_T1w.csv'
    t2_csaPath = 'csa-SC_T2w.csv'
    
    #Get data frame of subject eid and csa for T1w and T2w
    t1_csa =(get_csa(t1_csaPath)).set_index('Filename')
    t2_csa = (get_csa(t2_csaPath)).set_index('Filename')

    t1_col = list(csa_dict.values())[0][0]
    t2_col = list(csa_dict.values())[0][1]
    #Change the index to subject eid
    df = df.set_index('Subject')
    for sub in t1_csa.index:
        df.loc[int(sub), t1_col] = t1_csa.at[sub,'MEAN(area)']
        df.loc[int(sub), t2_col] = t2_csa.at[sub,'MEAN(area)']


    #5.Resahpe dataframe and write a csv file
    df =  df.drop(columns = ['Year of birth', 'Month of birth'])
    #print( df.head(5))
    filename = 'data_ukbiobank.csv'
    df.to_csv(filename)
    print(os.path.isfile(filename))

if __name__ == '__main__':
    main()
