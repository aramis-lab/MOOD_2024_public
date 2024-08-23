#!/bin/bash
#SBATCH --job-name=predict_vae_sobel
#SBATCH --output=logs/predict_vae_sobel/%j.out
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=01:00:00

echo $(date)

MOOD_IN=$1
MOOD_OUT=$2

ROOT_DIR=/workspace
VAE_DIR=${ROOT_DIR}/maps_abdom
JSON=extract_mood
DATASET=toy
SPLIT=0
MEAN_PATH=${MAPS}/split-${SPLIT}/residual_validation_mean.nii.gz
STD_PATH=${MAPS}/split-${SPLIT}/residual_validation_std.nii.gz


export PYTHONPATH=${PYTHONPATH}/${ROOT_DIR}

ln -s ${MOOD_OUT}/tmp /tmp

python ${ROOT_DIR}/main_pixel_abdom_vae.py --input $MOOD_IN --output $MOOD_OUT --maps $VAE_DIR --extract_json $JSON --dataset $DATASET --split $SPLIT --mean_path $MEAN_PATH --std_path $STD_PATH

CAPS_OUTPUT=${MOOD_OUT}/tmp/maps_abdom/split-${SPLIT}/best-loss/CapsOutput
LABEL_TSV=${CAPS_OUTPUT}/subjects_sessions.tsv
GEN_DIR=${ROOT_DIR}/maps_unet_abdom
PREPROCESSING_JSON=${ROOT_DIR}/MS_extract.json
MODE=abdom
SPLIT_GEN=0

MEAN_PATH_GEN=${ROOT_DIR}/maps_unet_abdom/split-${SPLIT_GEN}/mean_final_res_output_1_0.2.nii.gz
STD_PATH_GEN=${ROOT_DIR}/maps_unet_abdom/split-${SPLIT_GEN}/std_final_res_output_1_0.2.nii.gz
MEDIAN_PATH_GEN=${MEAN_PATH_GEN}

VAE_DIR=${MOOD_OUT}/tmp/maps_abdom
CAPS_PATH=${MOOD_OUT}/tmp/caps_brain

python ${ROOT_DIR}/mood/GAN/GAN_predict/predict_VAE_sobel_MS.py --caps $CAPS_PATH --json $PREPROCESSING_JSON --tsv $LABEL_TSV --maps_gen $GEN_DIR --vae_split $SPLIT --gen_split $SPLIT_GEN --mode $MODE --vae_dir $VAE_DIR --mean_path $MEAN_PATH_GEN --median_path $MEDIAN_PATH_GEN


NEW_CAPS_OUTPUT=${ROOT_DIR}/maps_unet_abdom/split-${SPLIT_GEN}/best-loss/CapsOutput

python ${ROOT_DIR}/main_pixel_abdom_postprocess.py --caps_output $NEW_CAPS_OUTPUT --tsv $LABEL_TSV --mean_path $MEAN_PATH_GEN --std_path $STD_PATH_GEN --input $MOOD_IN --output $MOOD_OUT

