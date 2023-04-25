# Cord CSA on UK biobank brain MRI database
This repository contains the analysis pipeline to normalize spinal cord cross-sectional area (CSA).

👉 Please cite this work if you use it or if you are inspired by it:
~~~
Bédard S, Cohen-Adad J. Automatic measure and normalization of spinal cord cross-sectional area using the pontomedullary junction. Frontiers in Neuroimaging [Internet]. 2022;1. Available from: https://www.frontiersin.org/articles/10.3389/fnimg.2022.1031253
~~~

Spinal cord CSA was computed on T1w MRI scans for 804 participants from the UK Biobank database. In addition to computing cross-sectional at the C2-C3 vertebral disc, it was also measured at 64 mm caudal from the pontomedullary junction (PMJ). The effect of various biological, demographic and anatomical factors was explored by computing Pearson’s correlation coefficients. A stepwise linear regression found significant predictors; the coefficients of the best fit model were used to normalize CSA.

Measure of the averaged cross-sectional area (CSA) between C2 and C3 of the spinal cord with UK Biobank Brain MRI dataset.
# Table of contents 
* [Data collection and organization](#data-collection-and-organization)
    * [Uk Biobank database](#uk-biobank-database)
    * [Data conversion: DICOM to BIDS](#data-conversion-dicom-to-bids)
    * [Acquisition parameters](#acquisition-parameters-todo-to-complete)
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

*The analysis pipeline was applied on T1w images only.*

The raw images have gradient distortion, correction will be applied in the preprocessing steps of the analysis pipeline. 

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
                ├── sub-1000710_T1w_pmj-manual.nii.gz  <------- manual pmj label
                ├── sub-1000710_T1w_pmj-manual.json
                ├── sub-1000710_T2w_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                └── sub-1000710_T2w_seg-manual.json
 
~~~

### Acquisition parameters
Scanner: Siemens Skyra 3T running VD13A SP4 with a standard Siemens 32-channel RF receive head coil
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

* [SCT 5.3.0](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/5.3.0) for processing
* [gradunwarp](gradunwarp_installation.md) for gradient correction
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
#### Note on gradient distortion correction
A `coeff.grad` associated with the MRI scanner used for the data is necessary if it has not been applied yet. In this project, the gradient distortion correction is done in `preprocess_data.sh` with `gradunwarp` and Siemens `coeff.grad` file.
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
Processing will generate spinal cord segmentation, vertebral labels, pmj label and compute cord CSA. Specify the path of preprocessed dataset with the flag `path-data`.

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
After running the analysis, check your Quality Control (qc) report by opening the file `~/ukbiobank_results/qc/index.html`. Use the "search" feature of the QC report to quickly jump to segmentations or labeling issues.

#### 1. Assess quality of segmentation, vertebral and pontomedullary junction (PMJ) labeling
If segmentation or labeling issues are noticed while checking the quality report, proceed to manual correction using the procedure below:

1. In QC report, search for "deepseg" to only display results of spinal cord segmentation, search for "vertebrae" to only display vertebral labeling and "pmj" to only display PMJ labeling.
2. Review segmentation and spinal cord and PMJ labeling, note that the segmentation et vertebral labeling need to be accurate only between C2-C3, for cord CSA. 
3. Click on the `F` key to indicate if the segmentation/label is OK ✅, needs manual correction ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, one for manual corrections and one for the unusable data, will automatically be generated. 
4. Download the lists by clicking on `Download QC Fails` and on `Download Qc Artifacts`. 

***Note: Proceed to QC separately for cord segmentation and vertebral labeling to generate 2 separate lists.***

The lists will have the following format:

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
*.yml list for correcting pontomedullary junction (PMJ) labeling:*
~~~
FILES_PMJ:
 - sub-1000032_T1w.nii.gz
 - sub-1000710_T1w.nii.gz
 ~~~

* `FILES_SEG`: Images associated with spinal cord segmentation
* `FILES_LABEL` Images associated with vertebral labeling
* `FILES_PMJ` Images associated with PMJ labeling

For the next steps, the script `uk_manual_correction` loops through all the files listed in .yml file and opens an interactive window to either manually correct segmentation, vertebral or PMJ labeling. Each manually-corrected label is saved under `derivatives/labels/` folder at the root of `PATH_DATA` according to the BIDS convention. Each manually-corrected file has the suffix `-manual`. The procedure is described bellow for cord segmentation and for vertebral labeling.

#### 2. Correct segmentations
For manual segmentation, you will need ITK-SNAP and this repository only. See **[installation](#installation)** instructions and **[dependencies](#dependencies)**.

Here is a tutorial for manually correcting segmentations. Note that the new QC report format with interactive features (✅/❌/⚠️) is not included in the tutorial.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/vCVEGmKKY3o/sddefault.jpg)](https://youtu.be/vCVEGmKKY3o "Correcting segmentations across multiple subjects")

Run the following line and specify the .yml list for cord segmentation with the flag `-config`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA>
~~~
After all corrections are done, you can generate a QC report by adding the flag `-qc-only-` to the command above. Note that SCT is required for generating QC report.

#### 3. Vertebral labeling
Note that manual labeling uses SCT and the QC report is generated automatically.

Here is a tutorial for manual vertebral labeling:

[![IMAGE ALT TEXT](http://img.youtube.com/vi/ycUrm97nW1A/sddefault.jpg)](https://youtu.be/ycUrm97nW1A "Correcting vertebral labeling across multiple subjects")

Run the following line and specify the .yml list for vertebral labeling with the flag `-config`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA>
~~~

To create disc labels, click at the posterior tip of the disc for C1-C2, C2-C3 and C3-C4 as shown in the following image: 

![readme_labels](https://user-images.githubusercontent.com/71230552/111220077-18fdf680-85af-11eb-8ec4-4774db842a27.PNG)

#### 4. PMJ labeling
Note that manual PMJ labeling uses SCT and the QC report is generated automatically.

Run the following line and specify the .yml list for PMJ labeling with the flag `-config`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA>
~~~
To create PMJ label, click at the posterior tip of the pontomedullary junction (PMJ) as shown in the following image:

![image](https://user-images.githubusercontent.com/71230552/125302462-f6c87b00-e2f9-11eb-9f78-79a4462a9aaa.png)

See [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3457) for more examples.

#### Upload the manually-corrected files

A QC report of the manually corrected files is created in a zip file. To update the dataset, add all manually-corrected files `derivatives/labels/`,  and include the qc zip file in the body of the PR. See our [internal procedure](https://github.com/neuropoly/data-management/blob/master/internal-server.md#upload) for more details. 

#### Add automatic segmentations to `derivatives/` folder
After all segmentations are manually QC-ed, you can add them to the `derivatives/` by running again the script `manual_correction.py` and adding the flag `-add-seg-only`:
~~~
uk_manual_correction -config <.yml file> -path-in ~/ukbiobank_results/data_processed -path-out <PATH_DATA> -add-seg-only
~~~
The automatic segmentations that did not require manual correction (files in .yml file) are added to the `derivatives/` folder. You can upload the folder following the instructions specified in [Upload the manually-corrected files](#upload-the-manually-corrected-files).

#### Re-run the analysis
After all the necessary segmentation and labels are corrected, re-run the analysis (`sct_run_batch` command in [Processing](#processing)). If manually-corrected files exist, they will be used instead of proceeding to automatic segmentation and labeling. Make sure to put the output results in another folder (flag `-path-output`) if you don't want the previous results to be overwritten. 

- - -

### Statistical analysis
#### Generate datafile:
To generate a data file with the CSA results from `process_data.sh` and a .csv file including the fields from the UK Biobank dataset, run the follwing line:

~~~
uk_get_subject_info -path-data <PATH_DATA> -path-output ~/ukbiobank_results/ -datafile <FILENAME>
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
    ├── csa-SC_c2c3.csv
    ├── csa-SC_pmj.csv
    ├── data_ukbiobank.csv
    └── stats_results
        ├── metrics
        |   ├── comparasion_c2c3_pmj
        |   |   └── scatterplots_c2c3_pmj_csa.png
        |   ├── scatter_plots
        |   |   ├── Age.jpeg
        |   |   ├── Brain GM volume.png
        |   |   ├── Brain WM volume.png
        |   |   ...
        |   ├── corr_table.csv
        |   ├── corr_table_and_pvalue.csv
        |   ├── corr_table_pvalue.csv        
        |   ├── stats_csa.csv
        |   ├── scatter_plot_age_brain_vol.png
        |   ├── scatter_plot_age_thalamus_vol.png
        |   └── stats_param.csv
        └── models
            ├── age
            |   ├── coeff
            |   |  ├── coeff_linear_fit.csv
            |   |  └── coeff_quadratic_fit.csv
            |   ├── summary
            |   |   ├── summary_linear_fit.txt
            |   |   └── summary_quadratic_fit.txt
            |   └── quadratic_fit.png
            ├── model_1
            |   └── CSA_PMJ
            |      ├── coeff
            |      |   ├── coeff_fullLin_CSA_PMJ.csv
            |      |   └── coeff_stepwise_CSA_PMJ.csv
            |      ├── residuals
            |      |    ├── res_plots_fullLin_CSA_PMJ.png
            |      |    └── res_plots_stepwise_CSA_PMJ.png
            |      ├── summary
            |      |   ├── summary_fullLin_CSA_PMJ.txt
            |      |   └── summary_stepwise_CSA_PMJ.txt
            |      └── compared_models.csv
            ├── model_2
            |  └── ...
            ├── sex
            |   └── violin_plot.csv
            └──norm_COV.csv
            
~~~
The coefficient files (`coeff_*.csv`) are included in [SCT](https://spinalcordtoolbox.com/) to normalize CSA.
