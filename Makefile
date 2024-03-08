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
    -DCMAKE_TOOLS_ADDRESS_SANITIZER=${CMAKE_TOOLS_ADDRESS_SANITIZER}

FILES_TO_LINT := $(shell find . \( -name '*.h' -or -name '*.cpp' -or -name '*.cc' \) \
                    -not -path '*/build/*' -not -path '*/install/*' -not -path '*/log/*')

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

.PHONY: test
test: build
    colcon --log-base /dev/null test \
            --ctest-args tests --symlink-install \
            --executor parallel --parallel-workers $$(nproc) \
            --event-handlers console_cohesion+

.PHONY: build-select
# packages="first_pkg second_pkg third_pkg..."
build-select:
    source ${ROS_ROOT}/setup.sh
    colcon --log-base /dev/null build \
        --base-paths packages \
        --symlink-install \
        --packages-up-to $(packages) \
        --cmake-args ${CMAKE_ARGS}

.PHONY: test-select
test-select:build-select
    colcon --log-base /dev/null test --ctest-args tests --symlink-install \
        --executor parallel --parallel-workers $$(nproc) \
        --event-handlers console_cohesion+ --packages-select $(packages)

.PHONY: lint-all
# args="--fix ..."
lint-all: build
    clang-tidy -p=build $(args) $(FILES_TO_LINT)

.PHONY: lint-one
lint-one: build
    clang-tidy -p=build $(args) $(FILES_TO_LINT)

.PHONY: clean
clean:
    rm -rf build install
