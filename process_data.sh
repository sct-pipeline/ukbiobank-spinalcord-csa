#!/bin/bash
#
# Process data.
#
# Usage:
#   ./process_data.sh <SUBJECT>
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
PATH_GRADCORR_FILE=$2 #to remove

# get starting time:
start=`date +%s`

# FUNCTIONS
# ==============================================================================

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels"
  #FILESEGLABELED="${file_seg}_labeled" to remove
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t1
  fi
}

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

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1 "t1"
file_t1_seg=$FILESEG

# Create disc label at C1-C2, C2-C3 and C3-C4 (only if it does not exist) 
label_if_does_not_exist ${file_t1} ${file_t1_seg}
file_t1_seg_labeled="${file_t1_seg}_labeled"

# Generate QC report to assess vertebral labeling
sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
# Flatten scan along R-L direction (to make nice figures)
sct_flatten_sagittal -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz
# Compute average cord CSA between C2 and C3
sct_process_segmentation -i ${file_t1_seg}.nii.gz -vert 2:3 -vertfile ${file_t1_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa-SC_T1w.csv -append 1

# T2w FLAIR
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"

# Segment spinal cord (only if it does not exist)
# Note: we specify the "t1" contrast for the automatic segmentation because the T2-FLAIR contrast is more similar to the T1 MPRAGE (this is due to the inversion recovery 'IR' in 'FLAIR' pulse which nulls the CSF signal)
segment_if_does_not_exist $file_t2 "t1"
file_t2_seg=$FILESEG
# Flatten scan along R-L direction (to make nice figures) 
sct_flatten_sagittal -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz

# Dilate t2 cord segmentation to use as mask for registration
file_t2_mask="${file_t2_seg}_dil"
ImageMath 3 ${file_t2_mask}.nii.gz MD ${file_t2_seg}.nii.gz 40

# Register T1w image to T2w FLAIR (rigid)
isct_antsRegistration -d 3 -m CC[ ${file_t2}.nii.gz , ${file_t1}.nii.gz , 1, 4] -t Rigid[0.5] -c 50x20x10 -f 8x4x2 -s 0x0x0 -o [_rigid, ${file_t1}_reg.nii.gz] -v 1 -x ${file_t2_mask}.nii.gz

# Apply transformation to T1w vertebral level
isct_antsApplyTransforms -i ${file_t1_seg_labeled}.nii.gz -r ${file_t2}.nii.gz -t _rigid0GenericAffine.mat -o ${file_t2}_seg_labeled.nii.gz -n NearestNeighbor

# Generate QC report to assess T1w registration to T2w
sct_qc -i ${file_t1}_reg.nii.gz -s ${file_t2}_seg_labeled.nii.gz -d ${file_t2}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Generate QC report to assess T2w vertebral labeling
sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_seg_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Compute average cord CSA between C2 and C3
sct_process_segmentation -i ${file_t2_seg}.nii.gz -vert 2:3 -vertfile ${file_t2}_seg_labeled.nii.gz -o ${PATH_RESULTS}/csa-SC_T2w.csv -append 1

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${SUBJECT}_T1w_seg.nii.gz" 
  "${SUBJECT}_T2w_seg.nii.gz"
  "${SUBJECT}_T1w_seg_labeled.nii.gz"
  "${SUBJECT}_T2w_seg_labeled.nii.gz"
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
