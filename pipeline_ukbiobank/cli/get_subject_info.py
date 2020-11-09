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

def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects info and writes it in data.csv file",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-subject', required=True, type=str,
                        help="Name of the subject.")
    parser.add_argument('-datafile', required=True, type=str,
                        help="Name of the csv file of the data.")
    parser.add_argument('-path-data-output', required=True, type=str,
                        help="Name output file of images.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    current = os.getcwd()

    #Get subject number
    subject = args.subject
    path_data_output = args.path_data_output
    os.chdir(path_data_output)
    #Create a new line in data file if it exists, adds subjects info, else, creates the file and adds subject info
    if (os.path.exists(args.datafile)):
        with open(args.datafile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([subject])
    else:
        with open(args.datafile, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Subject","CSA_T1","CSA_T2", "Intracranial Volume", "Age", "Sex", "Height","Weight" ])
                writer.writerow([subject])
if __name__ == '__main__':
    main()
