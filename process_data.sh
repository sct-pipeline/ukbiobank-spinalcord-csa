#!/bin/bash
#
# Process data shorten: proceeds to reorientation to RPI, resampling, gradient distortion correction and spinal cord segmentation.
#
# Usage:
#   ./process_data.sh <SUBJECT> <PATH_GRADCORR_FILE>
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
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

# FUNCTIONS
# ==============================================================================
# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

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
# Reorient to RPI and resample to 1 mm iso (supposed to be the effective resolution)
sct_image -i ${file_t1}.nii.gz -setorient RPI -o ${file_t1}_RPI.nii.gz
sct_resample -i ${file_t1}_RPI.nii.gz -mm 1x1x1 -o ${file_t1}_RPI_r.nii.gz
file_t1="${file_t1}_RPI_r"

# Gradient distorsion correction
gradient_unwarp.py ${file_t1}.nii.gz ${file_t1}_gradcorr.nii.gz siemens -g ${PATH_GRADCORR_FILE}/coeff.grad -n
file_t1="${file_t1}_gradcorr"

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1 "t1"


# T2
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"
# Reorient to RPI and resample to 1mm iso (supposed to be the effective resolution)
sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}_RPI.nii.gz
sct_resample -i ${file_t2}_RPI.nii.gz -mm 1x1x1 -o ${file_t2}_RPI_r.nii.gz
file_t2="${file_t2}_RPI_r"

#Gradient distorsion correction
gradient_unwarp.py ${file_t2}.nii.gz ${file_t2}_gradcorr.nii.gz siemens -g ${PATH_GRADCORR_FILE}/coeff.grad -n
file_t2="${file_t2}_gradcorr"

# Segment spinal cord (only if it does not exist)
# Note: we specify the "t1" contrast for the automatic segmentation because the T2-FLAIR contrast is more similar to the T1 MPRAGE (this is due to the inversion recovery 'IR' in 'FLAIR' pulse which nulls the CSF signal)
segment_if_does_not_exist $file_t2 "t1"

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${SUBJECT}_T1w_RPI_r_gradcorr.nii.gz"
  "${SUBJECT}_T2w_RPI_r_gradcorr.nii.gz"
  "${SUBJECT}_T1w_RPI_r_gradcorr_seg.nii.gz" 
  "${SUBJECT}_T2w_RPI_r_gradcorr_seg.nii.gz"
  
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
