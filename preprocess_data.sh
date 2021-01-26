#!/bin/bash
#
# Preprocess data.
# Usage:
#   ./preprocess_data.sh <SUBJECT> <PATH_GRADCORR_FILE>
#
#
# Authors: Sandrine Bédard, Julien Cohen-Adad
set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_GRADCORR_FILE=$2 

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
# Copy list of participants in resutls folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv $PATH_RESULTS/"participants.tsv"
fi
# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/

# T1w
# ------------------------------------------------------------------------------
file_t1="${SUBJECT}_T1w"
# Rename the raw image
mv ${file_t1}.nii.gz ${file_t1}_raw.nii.gz
file_t1="${file_t1}_raw"

# Reorient to RPI and resample to 1 mm iso (supposed to be the effective resolution)
sct_image -i ${file_t1}.nii.gz -setorient RPI -o ${file_t1}_RPI.nii.gz
sct_resample -i ${file_t1}_RPI.nii.gz -mm 1x1x1 -o ${file_t1}_RPI_r.nii.gz
file_t1="${file_t1}_RPI_r"

# Gradient distorsion correction
gradient_unwarp.py ${file_t1}.nii.gz ${file_t1}_gradcorr.nii.gz siemens -g ${PATH_GRADCORR_FILE}/coeff.grad -n
file_t1="${file_t1}_gradcorr"

# Rename gradcorr file
mv ${file_t1}.nii.gz ${SUBJECT}_T1w.nii.gz
# Delete raw, resampled and reoriented to RPI images
rm -f ${SUBJECT}_T1w_raw.nii.gz ${SUBJECT}_T1w_raw_RPI.nii.gz ${SUBJECT}_T1w_raw_RPI_r.nii.gz # Remove f or not?

# T2w FLAIR
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"
# Rename raw file
mv ${file_t2}.nii.gz ${file_t2}_raw.nii.gz
file_t2="${file_t2}_raw"

# Reorient to RPI and resample to 1mm iso (supposed to be the effective resolution)
sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}_RPI.nii.gz
sct_resample -i ${file_t2}_RPI.nii.gz -mm 1x1x1 -o ${file_t2}_RPI_r.nii.gz
file_t2="${file_t2}_RPI_r"

# Gradient distorsion correction
gradient_unwarp.py ${file_t2}.nii.gz ${file_t2}_gradcorr.nii.gz siemens -g ${PATH_GRADCORR_FILE}/coeff.grad -n
file_t2="${file_t2}_gradcorr"

# Rename gradcorr file
mv ${file_t2}.nii.gz ${SUBJECT}_T2w.nii.gz
# Delete raw, resampled and reoriented to RPI images
rm -f ${SUBJECT}_T2w_raw.nii.gz ${SUBJECT}_T2w_raw_RPI.nii.gz ${SUBJECT}_T2w_raw_RPI_r.nii.gz # Remove f or not?

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${SUBJECT}_T1w.nii.gz"
  "${SUBJECT}_T2w.nii.gz"
  
)
pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${SUBJECT}/anat/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
