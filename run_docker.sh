#!/usr/bin/env bash

DATASETS="/beegfs/datasets/"

# Argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS=$(realpath "${2}")
            shift
            shift
        ;;
        *)
            echo "ERROR: Unknown option $arg"
            exit 1
        ;;
    esac
done

docker run --gpus all \
-it --rm --ipc=host \
-v ${PWD}:/workspace/BUSCA \
-v ${DATASETS}:/beegfs/datasets \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
--name busca_container \
busca:latest
