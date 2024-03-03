.SILENT:
.ONESHELL:
# https://stackoverflow.com/questions/2131213#comment133331794_60363121
.RECIPEPREFIX := $(.RECIPEPREFIX) $(.RECIPEPREFIX)

SHELL = /bin/bash
CMAKE_BUILD_TYPE ?= Release
CMAKE_TOOLS_ADDRESS_SANITIZER ?= OFF
CMAKE_ARGS ?= \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_TOOLS_ADDRESS_SANITIZER=${CMAKE_TOOLS_ADDRESS_SANITIZER} \

.PHONY: all
all:
    $(error Please use explicit targets)

.PHONY: build
build:
    source ${ROS_ROOT}/setup.sh
    colcon --log-base /dev/null build \
    --base-paths packages \
    --symlink-install \
    --cmake-args ${CMAKE_ARGS}

.PHONY: clean
clean:
    rm -rf build install
