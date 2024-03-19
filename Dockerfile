# ACHTUNG!
# Platform arm64 means nvidia jetson arm64v8.
# Image may be not compatible with other arm machines.

FROM --platform=linux/arm64v8 nvcr.io/nvidia/l4t-base:r35.1.0 AS handy-base-arm64

### INSTALL NVIDIA PACKAGES

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV FLAGS="-O3 -ffast-math -Wall -march=armv8.2-a+simd+crypto+predres -mtune=cortex-a57"

RUN apt-get update -q \
    && apt-get install -yq --no-install-recommends \
        nvidia-cuda-dev \
        nvidia-cudnn8-dev \
        nvidia-tensorrt-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# INSTALL JETSON STATS

RUN apt-get update -q \
    && apt-get install -yq --no-install-recommends python3-pip \
    && pip3 install --no-cache-dir -U pip \
    && pip3 install --no-cache-dir -U jetson-stats \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

FROM --platform=linux/amd64 ubuntu:20.04 AS handy-base-amd64

ENV FLAGS="-O3 -ffast-math -Wall"

FROM handy-base-${TARGETARCH} AS handy-common

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

### COMMON BASE

ENV CLANG_VERSION=12
ENV GCC_VERSION=9

RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
        apt-transport-https \
        apt-utils \
        gcc-${GCC_VERSION} \
        g++-${GCC_VERSION} \
        ca-certificates \
        clang-format-${CLANG_VERSION} \
        clang-tidy-${CLANG_VERSION} \
        cmake\
        git \
        gnupg2 \
        libboost-dev \
        libpython3-dev \
        make \
        software-properties-common \
        python3 \
        python3-dev \
        python3-pip \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV CC="gcc-${GCC_VERSION}"
ENV CXX="g++-${GCC_VERSION}"
ENV CFLAGS="${FLAGS}"
ENV CXXFLAGS="${FLAGS}"

### PREPARE FOR OPENCV

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        gfortran \
        file \
        libatlas-base-dev \
        libeigen3-dev \
        libjpeg-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libtbb-dev \
        libtbb2 \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV OPENCV_VERSION="4.8.0"

# PREPARE FOR TORCH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gfortran \
        libopenblas-dev \
        libopenmpi-dev \
        libomp-dev \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV TORCHVISION_VERSION=0.14.0

FROM handy-common AS handy-cuda-arm64

RUN wget -qO - https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && wget -qO - https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && cd opencv-${OPENCV_VERSION} && mkdir -p build && cd build \
    && OPENCV_MODULES=(core calib3d imgproc imgcodecs ccalib ximgproc \
        aruco cudev cudaarithm cudacodec cudafilters cudaimgproc) \
    && cmake .. \
        -DBUILD_LIST=$(echo ${OPENCV_MODULES[*]} | tr ' ' ',') \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DWITH_GTK=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCUDA_ARCH_BIN=5.3,6.2,7.2 \
        -DCUDA_ARCH_PTX= \
        -DCUDA_FAST_MATH=ON \
        -DCUDNN_INCLUDE_DIR=/usr/include \
        -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -DWITH_EIGEN=ON \
        -DENABLE_NEON=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUBLAS=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_LIBV4L=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_TBB=ON \
        -DWITH_TIFF=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_WITH_OPENJPEG=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_IMGCODEC_HDR=OFF \
        -DWITH_IMGCODEC_SUNRASTER=OFF \
        -DWITH_IMGCODEC_PXM=OFF \
        -DWITH_IMGCODEC_PFM=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
    && make -j$(nproc) install && rm -rf /tmp/*

RUN pip3 install --no-cache-dir \
    Cython \
    wheel \
    numpy \
    pillow

ENV PYTORCH_WHL="torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl"
ENV PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/${PYTORCH_WHL}"

RUN wget --no-check-certificate -qO ${PYTORCH_WHL} ${PYTORCH_URL} \
    && pip3 install --no-cache-dir ${PYTORCH_WHL} \
    && rm -rf /tmp/*

RUN wget -qO - https://github.com/pytorch/vision/archive/refs/tags/v${TORCHVISION_VERSION}.tar.gz | tar -xz \
    && cd vision-${TORCHVISION_VERSION} \
    && python3 setup.py install \
    && rm -rf /tmp/*

ENV PYTORCH_PATH="/usr/local/lib/python3.8/dist-packages/torch"
ENV LD_LIBRARY_PATH="${PYTORCH_PATH}/lib:${LD_LIBRARY_PATH}"

FROM handy-common AS handy-cuda-amd64

RUN wget -qO - https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && wget -qO - https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && cd opencv-${OPENCV_VERSION} && mkdir -p build && cd build \
    && OPENCV_MODULES=(core calib3d imgcodecs ccalib ximgproc aruco) \
    && cmake .. \
        -DBUILD_LIST=$(echo ${OPENCV_MODULES[*]} | tr ' '  ',') \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DWITH_GTK=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -DWITH_EIGEN=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_GSTREAMER=OFF \
        -DWITH_LIBV4L=OFF \
        -DWITH_OPENCL=ON \
        -DWITH_IPP=OFF \
        -DWITH_TBB=ON \
        -DWITH_TIFF=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_WITH_OPENJPEG=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_IMGCODEC_HDR=OFF \
        -DWITH_IMGCODEC_SUNRASTER=OFF \
        -DWITH_IMGCODEC_PXM=OFF \
        -DWITH_IMGCODEC_PFM=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
    && make -j$(nproc) install && rm -rf /tmp/*

ENV TORCH_VERSION=1.13.0

RUN pip3 install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION}

ENV PYTORCH_PATH="/usr/local/lib/python3.8/dist-packages/torch"
ENV LD_LIBRARY_PATH="${PYTORCH_PATH}/lib:${LD_LIBRARY_PATH}"

FROM handy-cuda-${TARGETARCH} AS handy-ros

ENV ROS_VERSION=2
ENV ROS_DISTRO=iron
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

RUN wget -q https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

ENV RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"

RUN apt-get update -q \
    && apt remove -yq python-is-python2 \
    && apt-get install -yq --no-install-recommends \
        locales \
        python3-colcon-common-extensions \
        python3-flake8-docstrings \
        python-is-python3 \
        python3-pip \
        python3-pytest-cov \
        python3-rosinstall-generator \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN pip3 install --no-cache-dir -U \
   flake8-blind-except \
   flake8-builtins \
   flake8-class-newline \
   flake8-comprehensions \
   flake8-deprecated \
   flake8-import-order \
   flake8-quotes \
   "pytest>=5.3" \
   pytest-repeat \
   pytest-rerunfailures

ENV ROS_TMP=/tmp/${ROS_DISTRO}

RUN mkdir -p ${ROS_ROOT} \
    && mkdir -p ${ROS_TMP} && cd ${ROS_TMP} \
    && rosinstall_generator \
    --rosdistro ${ROS_DISTRO} \
    --deps \
        ament_cmake_clang_format \
        ament_cmake_clang_tidy \
        compressed_image_transport \
        cv_bridge \
        foxglove_bridge \
        foxglove_msgs \
        image_geometry \
        image_transport \
        geometry_msgs \
        geometry2 \
        launch_xml \
        launch_yaml \
        ros_base \
        rosbag2 \
        sensor_msgs \
        std_msgs \
        tf2 \
        vision_opencv \
        visualization_msgs \
    > ${ROS_ROOT}/ros2.rosinstall \
    && vcs import ${ROS_TMP} < ${ROS_ROOT}/ros2.rosinstall

RUN apt-get update -q \
    && rosdep init \
    && rosdep update \
    && rosdep install -qy --ignore-src  \
        --rosdistro ${ROS_DISTRO} \
        --from-paths ${ROS_TMP} \
        --skip-keys clang-format-${CLANG_VERSION} \
        --skip-keys clang-tidy-${CLANG_VERSION} \
        --skip-keys fastcdr \
        --skip-keys rti-connext-dds-6.0.1 \
        --skip-keys libopencv-dev \
        --skip-keys libopencv-contrib-dev \
        --skip-keys libopencv-imgproc-dev \
        --skip-keys python3-opencv \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

RUN cd ${ROS_TMP} \
    && colcon build \
        --merge-install \
        --install-base ${ROS_ROOT} \
        --cmake-args -DBUILD_TESTING=OFF \
    && rm -rf /tmp/*

RUN printf "export ROS_ROOT=${ROS_ROOT}\n" >> ${HOME}/.bashrc \
    && printf "export ROS_DISTRO=${ROS_DISTRO}\n" >> ${HOME}/.bashrc \
    && printf "export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}\n" >> ${HOME}/.bashrc \
    && printf "source ${ROS_ROOT}/setup.bash\n" >> ${HOME}/.bashrc

FROM handy-ros AS handy-dev

### INSTALL HIGH SPEED CAMERA SDK
RUN mkdir -p camera-sdk && cd camera-sdk \
    && wget -qO - https://storage.yandexcloud.net/the-lab-storage/handy/linuxSDK_V2.1.0.37.tar.gz | tar -xz \
    && ./install.sh \
    && rm -rf /tmp/*

### INSTALL DEV TOOLS

RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
        htop \
        less \
        lldb-${CLANG_VERSION} \
        tar \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV CFLAGS="${FLAGS} -std=c17"
ENV CXXFLAGS="${FLAGS} -std=c++2a"

RUN printf "export CC='${CC}'\n" >> ${HOME}/.bashrc \
    && printf "export CXX='${CXX}'\n" >> ${HOME}/.bashrc \
    && printf "export CFLAGS='${CFLAGS}'\n" >> ${HOME}/.bashrc \
    && printf "export CXXFLAGS='${CXXFLAGS}'\n" >> ${HOME}/.bashrc \
    && printf "export RCUTILS_LOGGING_BUFFERED_STREAM=1\n" >> ${HOME}/.bashrc \
    && printf "export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}:{time}] {message}'\n" >> ${HOME}/.bashrc \
    && ln -sf /usr/bin/clang-format-${CLANG_VERSION} /usr/bin/clang-format

WORKDIR /handy
ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["trap : TERM INT; sleep infinity & wait"]
