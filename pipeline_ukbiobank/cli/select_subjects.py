#!/usr/bin/env python
# -*- coding: utf-8
# Script to get the selected subject list ??
#
# For usage, type: select_subjects -h

import os
import argparse
import pipeline_ukbiobank
import csv
import yaml
import pandas as pd
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description="Genrates a selected subjects list (.yml) for the parameters who are available",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-subjects_datafile', required=True, type=str,
                        help="Name of the .csv file containing all the subject info.")

    return parser


def main ():
    parser = get_parser()
    args = parser.parse_args()
    #TODO read from the export_feilds_GBM3100
    sexe_id = "31"
    birth_month_id = "52"
    birth_year_id = "34"
    height_id = "12144-2.0"
    weight_id = "21002-2.0"
    icv_id = "25010-2.0"
    #Load subject data file
    dict_results = []
    with open(args.subjects_datafile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                dict_results.append(row)
    #Create a structure
    df = pd.DataFrame.from_dict(dict_results, orient='index')
    print(df[0:10])


if __name__ == '__main__':
    main()