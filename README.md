# Projet3
Mesure de l’aire de section médullaire sur la base de données UK Biobank

## Description

## Dependencies

SCT 5.0.0\
Python 3.7\
FSLeyes 

## Installation
Download this repository:
~~~
git clone https://github.com/sandrinebedard/Projet3.git
~~~
Install:
~~~
cd Projet3
pip install -e ./
~~~
## Usage
Create a folder where the results will be generated
~~~
mkdir ~/ukbiobank_results
~~~
Launch processing:
~~~
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ -script process_data.sh
~~~
