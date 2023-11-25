#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./execute.sh <camera ID>" >&2
    exit 2
fi

mkdir -p build
cmake -S . -B build
cmake --build build
./build/camera_identifier $1
