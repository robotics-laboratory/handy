FROM nvcr.io/nvidia/l4t-base:r35.1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

ENV ROS_VERSION=2
ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /tmp

### COMMON BASE

ENV CLANG_VERSION=12

RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
        apt-transport-https \
        apt-utils \
        ca-certificates \
        build-essential \
        clang-format-${CLANG_VERSION} \
        clang-tidy-${CLANG_VERSION} \
        cmake\
        curl \
        git \
        gnupg2 \
        libboost-dev \
        libpython3-dev \
        make \
        software-properties-common \
        gnupg \
        python3 \
        python3-dev \
        python3-distutils \
        python3-pip \
        python3-setuptools \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV CC="gcc-9"
ENV CXX="g++-9"
ENV FLAGS="-O3 -ffast-math -Wall -march=armv8.2-a+simd+crypto+predres -mtune=cortex-a57"
ENV CFLAGS="${FLAGS}"
ENV CXXFLAGS="${FLAGS}"

### INSTALL NVIDIA

RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
        nvidia-cuda-dev \
        nvidia-cudnn8-dev \
    && pip3 install --no-cache-dir -U jetson-stats==4.0.0 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

### INSTALL OPENCV

ARG OPENCV_VERSION="4.7.0"

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        gfortran \
        file \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libdc1394-22-dev \
        libeigen3-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        libjpeg-dev \
        libjpeg8-dev \
        libjpeg-turbo8-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtesseract-dev \
        libtiff-dev \
        libv4l-dev \
        libxine2-dev \
        libxvidcore-dev \
        libx264-dev \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

RUN wget -qO - https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && wget -qO - https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz | tar -xz \
    && cd opencv-${OPENCV_VERSION} && mkdir -p build && cd build \
    && OPENCV_MODULES=(core calib3d flann highgui imgcodecs video videoio bgsegm ccalib \
        cudev cudaarithm cudabgsegm cudacodec cudafilters cudaimgproc cudaoptflow cudev optflow ximgproc) \
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
        -DWITH_GSTREAMER=ON \
        -DWITH_LIBV4L=ON \
        -DWITH_OPENGL=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_TBB=ON \
        -DBUILD_TIFF=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
    && make -j$(nproc) install && rm -rf /tmp/*

### INSTALL LIBREALSENSE

ARG LIBRS_VERSION="2.53.1"

RUN apt-get update -yq \
    && apt-get install -yq --no-install-recommends \
        ca-certificates \
        libssl-dev \
        libusb-1.0-0-dev \
        libusb-1.0-0 \
        libudev-dev \
        libomp-dev \
        pkg-config \
        sudo \
        udev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

RUN wget -qO - https://github.com/IntelRealSense/librealsense/archive/refs/tags/v${LIBRS_VERSION}.tar.gz | tar -xz \
    && cd librealsense-${LIBRS_VERSION} \
    && bash scripts/setup_udev_rules.sh \
    && mkdir -p build && cd build \
    && cmake .. \
        -DBUILD_EXAMPLES=false \
        -DCMAKE_BUILD_TYPE=release \
        -DFORCE_RSUSB_BACKEND=false \
        -DBUILD_WITH_CUDA=true \
        -DBUILD_WITH_OPENMP=true \
        -DBUILD_PYTHON_BINDINGS=false \
        -DBUILD_WITH_TM2=false \
    && make -j$(($(nproc)-1)) install \
    && rm -rf /tmp/*

### INSTALL ROS2

RUN wget -q https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list \

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
    --exclude librealsense2 \
    --deps \
        ament_cmake_clang_format \
        compressed_depth_image_transport \
        compressed_image_transport \
        cv_bridge \
        foxglove_bridge \
        foxglove_msgs \
        image_geometry \
        image_transport \
        image_rotate \
        geometry_msgs \
        geometry2 \
        launch_xml \
        launch_yaml \
        realsense2_camera \
        ros_base \
        rosbridge_suite \
        sensor_msgs \
        std_msgs \
        tf2 \
        vision_opencv \
        visualization_msgs \   
    > ${ROS_ROOT}/ros2.rosinstall \
    && vcs import ${ROS_TMP} < ${ROS_ROOT}/ros2.rosinstall > /dev/null

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
        --skip-keys librealsense2 \
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

RUN printf "export ROS_ROOT=${ROS_ROOT}\n" >> /root/.bashrc \
    && printf "export ROS_DISTRO=${ROS_DISTRO}\n" >> /root/.bashrc \
    && printf "export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}\n" >> /root/.bashrc \
    && printf "source ${ROS_ROOT}/setup.bash\n" >> /root/.bashrc

### INSTALL PYTORCH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libopenmpi-dev \
        libomp-dev \
        gfortran \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* apt-get clean

RUN pip3 install --no-cache-dir \
    Cython \
    wheel \
    numpy \
    pillow

ARG PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl"
ARG PYTORCH_WHL="torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl"

RUN wget --no-check-certificate -qO ${PYTORCH_WHL} ${PYTORCH_URL} \
    && pip3 install --no-cache-dir ${PYTORCH_WHL} \ 
    && rm -rf /tmp/*

ARG TORCHVISION_VERSION=0.11.0

RUN wget -qO - https://github.com/pytorch/vision/archive/refs/tags/v${TORCHVISION_VERSION}.tar.gz | tar -xz \
    && cd vision-${TORCHVISION_VERSION} \
    && python3 setup.py install \
    && rm -rf /tmp/*

ENV PYTORCH_PATH="/usr/local/lib/python3.8/dist-packages/torch"
ENV LD_LIBRARY_PATH="${PYTORCH_PATH}/lib:${LD_LIBRARY_PATH}"

### INSTALL DEV PKGS

RUN apt-get update -q \
    && apt-get install -yq --no-install-recommends \
        htop \
        lldb-${CLANG_VERSION} \
        less \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ENV CFLAGS="${FLAGS} -std=c17"
ENV CXXFLAGS="${FLAGS} -std=c++2a"

RUN printf "export CC='${CC}'\n" >> /root/.bashrc \
    && printf "export CXX='${CXX}'\n" >> /root/.bashrc \
    && printf "export CFLAGS='${CFLAGS}'\n" >> /root/.bashrc \
    && printf "export CXXFLAGS='${CXXFLAGS}'\n" >> /root/.bashrc \
    && printf "export RCUTILS_LOGGING_BUFFERED_STREAM=1\n" >> /root/.bashrc \
    && printf "export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}:{time}] {message}'\n" >> /root/.bashrc \
    && ln -sf /usr/bin/clang-format-${CLANG_VERSION} /usr/bin/clang-format

### SETUP ENTRYPOINT

COPY /entrypoint.bash /entrypoint.bash
ENTRYPOINT ["/entrypoint.bash"]
WORKDIR /handy