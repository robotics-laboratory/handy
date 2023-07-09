#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: ./start-container.sh <username> <project path> <image version>" >&2
    exit 2
fi

image="cr.yandex/crp8hpfj5tuhlaodm4dl/handy:jetson-$3"
docker pull $image
docker run \
    -dit \
    --privileged \
    --network=host \
    --name handy-$1 \
    --runtime=nvidia \
    -v /dev:/dev \
    -v $(readlink -f $2):/handy \
    $image

