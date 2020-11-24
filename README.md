# Projet3
Mesure de l’aire de section médullaire sur la base de données UK Biobank
## Data collection and organization
### Data conversion: DICOM to BIDS
Here is an example of the data structure:
~~~
uk_biobank
│
├── participants.tsv
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
The images from UK Biobank Brain MRI used for this project are :
 - T1_orig_defaced.nii.gz
 - T2_FLAIR_orig_defaced.nii.gz
 
 T1_orig_defaced.nii.gz in the BIDS standard sub-XXXXXXX_T1w.nii.gz
 T2_FLAIR_orig_defaced.nii.gz in the BIDS standard sub-XXXXXXX_T2w.nii.gz
#### T1-weighted structural imaging
    Resolution: 2.4x2.4x2.4 mm
    Field-of-view: 88x88x64 matrix
    Duration: 6 minutes (490 timepoints)
    TR: 0.735 s
    TE: 39ms
    GE-EPI with x8 multislice acceleration, no iPAT, flip angle 52◦, fat saturation
#### T2-weighted FLAIR structural imaging
    Resolution: 1.05x1x1 mm
    Field-of-view: 192x256x256 matrix
    Duration: 6 minutes
    3D SPACE, sagittal, in-plane acceleration iPAT=2, partial Fourier = 7/8, fat saturation, elliptical k-space scanning, prescannormalise

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
