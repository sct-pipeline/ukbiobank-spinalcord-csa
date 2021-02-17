# Cord CSA on UK biobank brain MRI database
Measure of the averaged cross-sectional area (CSA) between C2 and C3 of the spinal cord with UK Biobank Brain MRI dataset.
# Table of contents 
* [Data collection and organization](#data-collection-and-organization)
    * [Uk Biobank database](#uk-biobank-database)
    * [Data conversion: DICOM to BIDS](#data-conversion-dicom-to-bids)
    * [Aquisition parameters](#aquisition-parameters-todo-to-complete)
* [Analysis pipeline](#analysis-pipeline)
    * [Dependencies](#dependencies)
    * [Installation](#installation)
    * [Preprocessing](#preprocessing)
    * [Processing](#processing)
    * [Quality control](#quality-control)
    * [Statistical analysis](#statistical-analysis)
    
- - -
## Data collection and organization
### Uk Biobank database
The brain MRI data of UK biobank follows the DICOM convention. The spinal cord of the processed brain MRI images is cut off. Because the purpose of this project is to measure the CSA between C2 and C3 of the spinal cord, the raw MRI images for T1w structural images and for T2w FLAIR are used as an input of the pipeline:
 * `T1_orig_defaced.nii.gz`
 * `T2_FLAIR_orig_defaced.nii.gz`

The raw images have gradient distorsion, a distorsion correction will be applied in the preprocessing steps of the analysis pipeline. 

The DICOM dataset is under: `duke:mri/uk_biobank`
### Data conversion: DICOM to BIDS
For this project, a BIDS standard dataset is used. A conversion of DICOM to BIDS is necessary for the UK Biobank dataset. 
The data from the DICOM dataset in the BIDS standard for this project have the following correspondance for each subjects:
 * `T1_orig_defaced.nii.gz` in the BIDS standard is `sub-XXXXXXX_T1w.nii.gz`
 * `T2_FLAIR_orig_defaced.nii.gz` in the BIDS standard is`sub-XXXXXXX_T2w.nii.gz`
 
To convert the DICOM dataset in a BIDS structure for this project, run the following line:
~~~
curate_project.py -path-in <path_DICOM_dataset> -path-output <path_BIDS_dataset>
~~~
The BIDS datset with raw data is under: `data.neuro.polymtl.ca:datasets/uk-biobank`.
The dataset resulting from [preprocessing](#preprocessing) is under `data.neuro.polymtl.ca:datasets/uk-biobank-processed`.

Here is an example of the BIDS data structure of uk-biobank-processed:
~~~
uk-biobank-processed
│
├── dataset_description.json
├── participants.json
├── participants.tsv
├── README
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
                ├── sub-1000710_T1w_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                ├── sub-1000710_T1w_seg-manual.json  <------------ information about origin of segmentation
                ├── sub-1000710_T1w_labels-manual.nii.gz  <------- manual vertebral labels
                ├── sub-1000710_T1w_labels-manual.json
                ├── sub-10007106_T2w_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                └── sub-1000710_T2w_seg-manual.json
 
~~~

### Aquisition parameters |TODO to complete
--> add scanner
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
* [ANTs](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS) for processing
* Python 3.7  for statistical analysis

OPTIONAL:

* [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) for correcting cord segmentations

    **NOTE:** 
    Make sure to add ITK-SNAP to the system path:
    - For Windows, select the option during installation.
    - For macOS, after installation, go to **Help->Install Command-Line Tools**.

### Installation
Download this repository:
~~~
git clone https://github.com/sct-pipeline/ukbiobank-spinalcord-csa.git
~~~
Install:
~~~
cd ukbiobank-spinalcord-csa
pip install -e ./
~~~
#### Note on gradient distorsion correction
A `coeff.grad` associated with the MRI scanner used for the data is necessary if it has not been applied yet. In this project, the gradient distorsion correction is done in `preprocess_data.sh` with [gradunwrap v1.2.0](https://github.com/Washington-University/gradunwarp/tree/v1.2.0) and Siemens `coeff.grad` file.
- - -
### Preprocessing
Preprocessing generates a dataset with gradient distortion correction. 

First, initialize shell variable with the path to the folder with the `coeff.grad` file:
~~~
PATH_GRADCORR_FILE=<path-gradcorr>
~~~
Launch preprocessing:
~~~
sct_run_batch -jobs -1 -path-data <PATH-DATA> -path-output ~/ukbiobank_preprocess -script preprocess_data.sh -script-args $PATH_GRADCORR_FILE
~~~

The results to use as the new dataset wil be in `~/ukbiobank_preprocess/data_processed/`.

### Processing
Processing will generate spinal cord segmentation, vertebral labels and compute cord CSA. Specify the path of preprocessed dataset with the flag `path-data`.

Launch processing:
~~~
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ -script process_data.sh
~~~

Or you can launch processing with a config file instead by using the flag `-config` and by adjusting the file `config_sct_run_batch.yml` according to your setups.
See `sct_run_batch -h` to look at the available options. To launch processing:

~~~
sct_run_batch -config config_sct_run_batch.yml
~~~

- - -
### Quality control
After running the analysis, check your Quality Control (qc) report by opening the file `~/ukbiobank_results/qc/index.html`. Use the "search" feature of the QC report to quikly jump to segmentations or labeling issues.

#### 1. Assess quality of segmentation and vertebral labeling
If segmentation or labeling issues are noticed while checking the quality report, proceed to manual segmentation correction or manual labeling of C2-C3 intervertebral disc at the posterior tip of the disc using the procedure below:

1. Create two .yml files that list the data to correct, one for segmentation and the other for vertebral labeling.
2. In QC report, search for "deepseg" to only display results of spinal cord segmentation, search for "vertebrae" to only display vertebral labeling.
3. Review segmentation and spinal cord labeling, note that the segmentation et vertebral labeling need to be accurate only between C2-C3, for cord CSA. 
4. If *major* issues are detected for C2-C3 segmentation and vertebral labeling, add the image's name into the corresponding .yml file as in the example below:

*.yml list for correcting cord segmentation:*
~~~
FILES_SEG:
- sub-1000032_T1w.nii.gz
- sub-1000083_T2w.nii.gz
~~~
*.yml list for correcting vertebral labeling:*
~~~
FILES_LABEL:
- sub-1000032_T1w.nii.gz
- sub-1000710_T1w.nii.gz
~~~

* `FILES_SEG`: Images associated with spinal cord segmentation
* `FILES_LABEL` Images associated with vertebral labeling (T1w images only)

For the next steps, the script `uk_manual_correction` loops through all the files listed in .yml file and opens an interactive window to either manually correct segmentation or vertebral labeling. Each manually-corrected label is saved under `derivatives/labels/` folder at the root of `PATH_DATA` according to the BIDS convention. Each manually-corrected file has the suffix `-manual`. The procedure is described bellow for cord segmentation and for vertebral labeling.

#### 2. Correct segmentations
For manual segmentation, you will need ITK-SNAP and this repository only. See **[installation](#installation)** instructions and **[dependencies](#dependencies)**.

**TODO: put video here**

Run the following line and specify the .yml list for cord segmentation with the flag `-config`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA>
~~~
After all corrections are done, you can generate a QC report by adding the flag `-qc-only-` to the command above. Note that SCT is required for generating QC report.

#### 3. Vertebral labeling
Note that manual labeling uses SCT and the QC report is generated automatically.

**TODO: put video here**

Run the following line and specify the .yml list for vertebral labeling with the flag `-config`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA>
~~~

C2-C3 disc label will be located at the posterior tip of the disc as shown in the following image. 

![alt text](https://user-images.githubusercontent.com/2482071/100895704-dabf4a00-348b-11eb-8b1c-67d5024bfeda.png)

#### Upload the manually-corrected files

A QC report of the manually corrected files is created in a zip file. To update the dataset, add all manually-corrected files `derivatives/labels/`,  and include the qc zip file in the body of the PR. See our [internal procedure](https://github.com/neuropoly/data-management/blob/master/internal-server.md#upload) for more details.
**TODO see if it is possible to inlcude the qc zipe file in PR** 

#### Re-run the analysis
After all the necessary segmentation and labels are corrected, re-run the analysis (`sct_run_batch`command in [Processing](###processing)). If manually-corrected files exists, they will be used intead of proceeding to automatic segmentation and labeling. Make sure to put the output results in another folder (flag `-path-output`) if you don't want the previous relsults to be overwritten. 

- - -

### Statistical analysis
#### Generate datafile:
To generate a data file with the CSA results from `process_data.sh` and fields from `participant.tsv`, run the follwing line:

~~~
uk_get_subject_info -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ 
~~~

#### Compute statistical analysis
To compute the statistical analysis of cord CSA results, use `uk_compute_stats`.  If the datafile ouput of `uk_get_subject_info` is not `data_ukbiobank.csv`, add the flag `-dataFile <FILENAME>`. Run this script in `/results` folder or specify path to folder that contains output files of analysis pipeline in with `-path-output` flag. The flag -exclude points to a yml file containing the subjects to be excluded from the statistical analysis:
~~~
uk_compute_stats -path-output ~/ukbiobank_results/ -dataFile <DATAFILE> -exclude <EXCLUDE.yml>
~~~

The output of `uk_compute_stats`has the following data structure:
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
