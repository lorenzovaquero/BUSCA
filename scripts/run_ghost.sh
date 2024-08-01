#!/usr/bin/env bash

# Base arguments
TRACKER_NAME="GHOST"
EXP_DATE="$(date +"%Y-%m-%d_%H-%M-%S")"
EXP_NAME="${EXP_DATE}_${TRACKER_NAME}"

REPO_FOLDER=$(dirname "${0}")
REPO_FOLDER=$(realpath "${REPO_FOLDER}/..")

MAIN_SCRIPT_FOLDER="${REPO_FOLDER}/trackers/GHOST"
MAIN_SCRIPT="${MAIN_SCRIPT_FOLDER}/tools/main_track.py"


# Default arguments
DATASET="MOT17"  # "MOT17" | "MOT20"
TEST_SET="val"  # "val" | "test"
USE_BUSCA=true
BUSCA_CONFIG="${REPO_FOLDER}/config/GHOST/MOT17/config_ghost_mot17.yml"
BUSCA_CHECKPOINT="${REPO_FOLDER}/models/BUSCA/motsynth/model_busca.pth"
OUTPUT_BASE_FOLDER="${REPO_FOLDER}/exp"
ONLINE_VISUALIZATION=false


# Argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift
            shift
        ;;
        --testset)
            TEST_SET="$2"
            shift
            shift
        ;;
        --use-busca)
            USE_BUSCA=true
            shift
        ;;
        --online-visualization)
            ONLINE_VISUALIZATION=true
            shift
        ;;
        --busca-config)
            BUSCA_CONFIG="$2"
            shift
            shift
        ;;
        --busca-ckpt)
            BUSCA_CHECKPOINT="$2"
            shift
            shift
        ;;
        --output-base-folder)
            OUTPUT_BASE_FOLDER="$2"
            shift
            shift
        ;;
        *)
            echo "ERROR: Unknown option $arg"
            exit 1
        ;;
    esac
done


# Argument rewriting and checks
if [ "$USE_BUSCA" = true ] ; then
    use_busca="--use-busca"
    exp_name="${EXP_NAME}_BUSCA"
else
    use_busca=""
    exp_name="${EXP_NAME}_BASE"
fi


if [ "$ONLINE_VISUALIZATION" = true ] ; then
    online_visualization="--online-visualization"
else
    online_visualization=""
fi


if [ "${DATASET}" == "MOT17" ]; then
    ghost_config="${MAIN_SCRIPT_FOLDER}/config/config_tracker.yaml"
    dataset_args="--inact 0.7 --do_inact 1 --combi sum_0.6 --on_the_fly 1 --det_conf 0.5 --new_track_conf 0.55 --last_n_frames 90"
    if [ "${TEST_SET}" == "val" ]; then
        valset='mot17_train'
        det_file='yolox_dets_val.txt'
    elif [ "${TEST_SET}" == "test" ]; then
        valset='mot17_test'
        det_file='yolox_dets.txt'
    else
        echo "ERROR: testset must be 'val' or 'test' (got ${TEST_SET})"
        exit 1
    fi
elif [ "${DATASET}" == "MOT20" ]; then
    ghost_config="${MAIN_SCRIPT_FOLDER}/config/config_tracker_20.yaml"
    dataset_args="--inact 0.8 --combi sum_0.8 --det_conf 0.45 --new_track_conf 0.6 --last_n_frames 30 --len_thresh 0 --remove_unconfirmed 0"
    if [ "${TEST_SET}" == "val" ]; then
        valset='mot20_train'
        # det_file='yolox_dets_val.txt'  # Gives negative MOTAs. Probably is YOLOX trained on MOT17 val (and for MOT20 is not good)
        det_file='yolox_dets.txt'
    elif [ "${TEST_SET}" == "test" ]; then
        valset='mot20_test'
        det_file='yolox_dets.txt'
    else
        echo "ERROR: testset must be 'val' or 'test' (got ${TEST_SET})"
        exit 1
    fi
else
    echo "ERROR: dataset must be 'MOT17' or 'MOT20' (got ${DATASET})"
    exit 1
fi


output_dir="${OUTPUT_BASE_FOLDER}/${TRACKER_NAME}/${DATASET}/${TEST_SET}/${exp_name}"
results_dir="${output_dir}/track_results"
log_file="${output_dir}/out.txt"


# Echo info
echo "${exp_name}"
echo "Running tracker ${TRACKER_NAME} on dataset ${DATASET} (${TEST_SET} set)"
if [ "$USE_BUSCA" = true ] ; then
    echo "BUSCA is ENABLED"
    echo "Using config file: ${BUSCA_CONFIG}"
    echo "Using checkpoint: ${BUSCA_CHECKPOINT}"
else
    echo "BUSCA is DISABLED"
fi
echo "Saving results to: ${output_dir}"
if [ "$ONLINE_VISUALIZATION" = true ] ; then
    echo "Online visualization ENABLED"
fi


# Create folders
mkdir -p "${output_dir}"
mkdir -p "${results_dir}"
touch "${log_file}"


# Set command
command="PYTHONPATH=${MAIN_SCRIPT_FOLDER}:${PYTHONPATH} python3 ${MAIN_SCRIPT} --splits ${valset} --det_file ${det_file} --config_path ${ghost_config} \
  --datasets-dir ${MAIN_SCRIPT_FOLDER} --reid-weights-dir ${MAIN_SCRIPT_FOLDER} \
  --store_feats 0 --act 0.7 --inact_patience 50 ${dataset_args} \
  --output-dir=${output_dir} --ignore-ghost-exp-name \
  ${use_busca} --busca-config=${BUSCA_CONFIG} --busca-ckpt=${BUSCA_CHECKPOINT} ${online_visualization}"


# Log info
echo "${exp_name}" >> "${log_file}"
echo "" >> "${log_file}"
echo "${command}" 2>&1 | tee -a "${log_file}"
echo "" 2>&1 | tee -a "${log_file}"


# Run tracker
eval "$command" 2>&1 | tee -a "${log_file}"

# Replicate results for MOT17 (we aren't evaluating on public detections)
if [ "${DATASET}" == "MOT17" ]; then
    if [ "${test_set}" == "test" ]; then
        echo "Replicating result files..."
        # We only evaluated on the "-FRCNN" videos, so we need to copy the results to the other videos
        for res_file in "${results_dir}"/MOT17-*-FRCNN; do
            if [ -e "$res_file" ]; then
                base_name="${res_file%-FRCNN}"
                cp "$res_file" "${base_name}-DPM.txt"
                cp "$res_file" "${base_name}-SDP.txt"
                mv "$res_file" "${base_name}-FRCNN.txt"  # We also rename the original file because GHOST did not include the .txt extension
                echo "Created files: ${base_name}-DPM.txt and ${base_name}-SDP.txt"
            fi
        done
    fi
fi

# Log info
echo "" 2>&1 | tee -a "${log_file}"
echo "Experiment ${exp_name} done!" 2>&1 | tee -a "${log_file}"
echo "Results can be found at ${results_dir}" 2>&1 | tee -a "${log_file}"

