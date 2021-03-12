#!/usr/bin/env python

from setuptools import setup, find_packages
import pathlib
from os import path

here = pathlib.Path(__file__).parent.resolve()
# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

req_path = path.join(here, 'requirements.txt')
with open(req_path, "r") as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setup(
    name='pipeline_ukbiobank', 
    version='1.0',
    python_requires='>=3.7',
    description='Collection of cli to process data from cord CSA project on UK Biobank database', 
    url='https://github.com/sandrinebedard/Projet3', 
    #author='',  # Optional
    #author_email='author@example.com',  # Optional
    classifiers=[  
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='',
    install_requires=install_reqs,
    packages=find_packages(exclude=['.git', '.github', '.docs']),
    include_package_data=True,
    package_data={
        '': ['*.png', '*.json', '*.r'],
    },
    
    entry_points={
        'console_scripts': [
            'uk_get_subject_info = pipeline_ukbiobank.cli.get_subject_info:main',
            'uk_select_subjects = pipeline_ukbiobank.cli.select_subjects:main',
            'uk_compute_stats = pipeline_ukbiobank.cli.compute_stats:main',
            'uk_manual_correction = pipeline_ukbiobank.cli.manual_correction:main',
            'uk_package_for_correction = pipeline_ukbiobank.cli.package_for_correction:main'
        ],
    },

 
)
