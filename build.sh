#!/usr/bin/env bash

# Depending on your Docker runtime, you may need to run the script with `DOCKER_BUILDKIT=0 ./build.sh` (https://stackoverflow.com/a/75629058)
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f Dockerfile -t busca .
