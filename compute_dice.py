#!/usr/bin/env python
# -*- coding: utf-8
# Computes mean dice score accross manual segmentations from candidates and ground truth segmentation. 
#
# For usage, type: python compute_dice.py -h
#
# Authors: Sandrine Bédard

import argparse
import logging
import os
import sys
import yaml
import pandas as pd
from textwrap import dedent

def get_parser():
    parser = argparse.ArgumentParser(
        description=" TODO",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-path-ref',
                        required=True, 
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the derivative folder of the ground truth segmentations.")
    parser.add_argument('-path-seg', 
                        required=True,
                        type=str,
                        metavar='<dir_path>',
                        help="Path to the folder including all the candidates manual segmentations.")
    parser.add_argument('-path-output',
                        required=False,
                        type=str,
                        default='./', 
                        metavar='<filename>',
                        help="Path where results will be written.")
    return parser

def main():


if __name__ == '__main__':
    main()