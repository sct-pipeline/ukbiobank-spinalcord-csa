#!/usr/bin/env python
#
# Script to get the user info from .json file.
#
# For usage, type: get_subject_info -h

import os
import argparse
import spinegeneric as sg
import spinegeneric.utils
import csv
import json
import pandas as pd
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets the subjects info and writes it in data.csv file",
        formatter_class=sg.utils.SmartFormatter,
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-jsonfile', required=True, type=str,
                        help="Name of the .json file the subject.")
    parser.add_argument('-c', required=True, type=str,
                        help="Contraste. {t1,t2}")
    parser.add_argument('-datafile', required=True, type=str,
                        help="Name of the csv file of the data.")
    parser.add_argument('-path-data-output', required=True, type=str,
                        help="Name output file of iamges.")
    return parser
def get_manufacturer_Json(fileName):
    with open(fileName, "r") as f:
        data = json.load(f)
        Manufacturer = data["Manufacturer"]
    return Manufacturer

def main():
    parser = get_parser()
    args = parser.parse_args()
    current = os.getcwd()
    #read json
    Manufcaturer = get_manufacturer_Json(args.jsonfile)
    #Get subject number
    subject = args.jsonfile[:-9]
    subject_info = [subject, args.c, Manufcaturer]
    #add to data.csv file
    path_data_output = args.path_data_output
    os.chdir(path_data_output)
    #Create a new line in data file, adds subjects info
    with open(args.datafile, 'a', newline='') as file:
            writer = csv.writer(file)
            # Delete next line, for validtion only
            #writer.writerow(["Subject", "Contrast", "Manufacturer","CSA", "Intracranial Volume", "Age", "Sex", "Height","Weight" ])
            writer.writerow([subject,args.c,Manufcaturer])
if __name__ == '__main__':
    main()
