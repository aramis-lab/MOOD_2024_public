#!/bin/bash
#SBATCH --output=logs/docker/sample_brain_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=02:00:00

echo $(date)

ROOT_DIR=/workspace
MAPS=${ROOT_DIR}/maps
JSON=extract_mood
CUSTOM_SUFFIX=*mood*
DATASET=toy
SPLIT=0
MODE=sample
MEAN_PATH=${MAPS}/split-${SPLIT}/residual_validation_mean.nii.gz
STD_PATH=${MAPS}/split-${SPLIT}/residual_validation_std.nii.gz

MOOD_IN=$1
MOOD_OUT=$2

# cd $MOOD_OUT
# git clone https://github.com/aramis-lab/clinicadl.git
# chmod a+rwx -R $MOOD_OUT
# cd clinicadl 
# poetry install 
# git switch ms_caps_maps
# cd $ROOT_DIR
#ln -s ${MOOD_OUT}/tmp /tmp

python ${ROOT_DIR}/main_sample_brain.py --input $MOOD_IN --output $MOOD_OUT --maps $MAPS --extract_json $JSON --custom_suffix $CUSTOM_SUFFIX --dataset $DATASET --split $SPLIT --mode $MODE --mean_path $MEAN_PATH --std_path $STD_PATH

echo $(date)
