# Projet3
Measure of the averaged cross-sectional area (CSA) between C2 and C3 of the spinal cord with UK Biobank Brain MRI dataset.
## Description
- - -
## Data collection and organization
### Uk Biobank database
The brain MRI data of UK biobank follows the DICOM convention. The spinal cord of the processed brain MRI images is cut off. Because the purpose of this project is to measure the CSA between C2 and C3 of the spinal cord, the raw MRI images for T1w structural images and for T2w FLAIR are used as an input of the pipeline:
 * `T1_orig_defaced.nii.gz`
 * `T2_FLAIR_orig_defaced.nii.gz`

The raw images have gradient distorsion, a distorsion correction will be applied in the analysis pipeline. 

The DICOM dataset is in: `duke:mri/uk_biobank`
### Data conversion: DICOM to BIDS
For this project, a BIDS standard dataset is used. A conversion of DICOM to BIDS is necessary for the UK Biobank dataset. 
The data from the DICOM dataset in the BIDS standard for this project have the following correspondance for each subjects:
 * `T1_orig_defaced.nii.gz` in the BIDS standard is `sub-XXXXXXX_T1w.nii.gz`
 * `T2_FLAIR_orig_defaced.nii.gz` in the BIDS standard is`sub-XXXXXXX_T2w.nii.gz`
 
To convert the DICOM dataset in a BIDS structure for this project, run the following line:
~~~
curate_project.py -path-in <path_DICOM_dataset> -path-output <path_BIDS_dataset>
~~~
The BIDS datset is (for now) in `duke: temp/uk_biobank_BIDS`

Here is an example of the BIDS data structure of UK biobank data:
~~~
uk_biobank_BIDS
│
├── participants.tsv
├── subjects_gbm3100.csv
├── sub-1000032
├── sub-1000083
├── sub-1000252
├── sub-1000498
├── sub-1000537
├── sub-1000710
│   │
│   └── anat
│       ├── sub-1000710_T1w.json
│       ├── sub-1000710_T1w.nii.gz
│       ├── sub-1000710_T2w.json
│       └── sub-1000710_T2w.nii.gz
└── derivatives
    │
    └── labels
        └── sub-1000710
            │
            └── anat
                ├── sub-1000710_T1w_RPI_r_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                ├── sub-1000710_T1w_RPI_r_seg-manual.json  <------------ information about origin of segmentation (see below)
                ├── sub-1000710_T1w_RPI_r_labels-manual.nii.gz  <------- manual vertebral labels
                ├── sub-1000710_T1w_RPI_r_labels-manual.json
                ├── sub-10007106_T2w_RPI_r_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                └── sub-1000710_T2w_RPI_r_seg-manual.json
 
~~~

### Aquisition parameters
#### T1-weighted structural imaging
    Resolution: 1x1x1 mm
    Field-of-view: 208x256x256 matrix
#### T2-weighted FLAIR structural imaging
    Resolution: 1.05x1x1 mm
    Field-of-view: 192x256x256 matrix
- - -
## Analysis pipeline
This repository includes a collection of scripts to analyse a BIDS-structured MRI dataset.

### Dependencies
MANDATORY:

* [SCT 5.0.1](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/5.0.1) for processing
* [gradunwrap v1.2.0](https://github.com/Washington-University/gradunwarp/tree/v1.2.0) for gradient correction
* Python 3.7  for statistical analysis

OPTIONAL:

* [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes) for correcting segmentations

### Installation
Download this repository:
~~~
git clone https://github.com/sandrinebedard/Projet3.git
~~~
Install:
~~~
cd Projet3
pip install -e ./
~~~
### Note on gradient distorsion correction
A `coeff.grad` associated with the MRI used for the data is necessary if it has not been applied yet. In this project, the gradient distorsion correction is done in `process_data.sh` with [gradunwrap v1.2.0](https://github.com/Washington-University/gradunwarp/tree/v1.2.0) and Siemens `coeff.grad` file.
- - -
### Usage
Create a folder where the results will be generated
~~~
mkdir ~/ukbiobank_results
~~~
Initialize shell variable with the path to the folder with the `coeff.grad` file:
~~~
PATH_GRADCORR_FILE=<path-gradcorr>
~~~
Launch processing:
~~~
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ -script process_data.sh -script-args $PATH_GRADCORR_FILE
~~~
- - -
### Statistical analysis
#### Generate datafile:
To generate a data file with the CSA results from `process_data.sh`, run the follwing line:

~~~
uk_get_subject_info -path-data <PATH_DATA> -path-output ~/ukbiobank_results/
~~~

If in the initial BIDS data structure, the file subjects_gbm3100.csv with fields of the subjects has another name, specify the name using the flag `-datafile`:

~~~
uk_get_subject_info -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ -datafile <FILENAME>
~~~
#### Compute statistical analysis
To compute the statistical analysis of the data of the pipeline analysis with the output file of `uk_get_subject_info`, use `uk_compute_stats`.  If the datafile ouput of uk_get_subject_info is not `data_ukbiobank.csv`, add the flag `-dataFile <FILENAME>`. Run this script in `/results` folder or specify path to folder that contains output files in with `-path-results` flag. The flag -exclude points to a yml file containing the subjects to be excluded from the statistics:
~~~
uk_compute_stats -path-results ~/ukbiobank_results/results -dataFile <DATAFILE> -exclude <EXCLUDE.yml>
~~~

The output of `uk_compute_stats`has the following data_structure:
~~~
ukbiobank_results
│
├── data_processed
├── log
├── qc
└── results
    ├── csa-SC_T1w.csv
    ├── csa-SC_T2w.csv
    ├── data_ukbiobank.csv
    ├── log_stats
    └── stats_results
        ├── metrics
        |   ├── corr_table.csv
        |   ├── stats_csa.csv
        |   └── stats_param.csv
        └── models
            ├── T1w_CSA
            └── T2w_CSA
                ├── coeff
                |   ├── coeff_fullLin_T2w_CSA.csv
                |   └── coeff_stepwise_T2w.CSA
                ├── residuals
                |    ├── res_plots_fullLin_T2w_CSA
                |    └── res_plots_stepwise_T2w_CSA
                ├── summary
                |   ├── summary_fullLin_T2w_CSA
                |   └── summary_stepwise_T2w_CSA
                └── compared_models.csv
