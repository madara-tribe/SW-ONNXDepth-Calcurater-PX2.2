FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build
COPY install_onnx_runtime.bash  /tmp/install_onnx_runtime.bash 
COPY install_apps_dependencies.bash /tmp/install_apps_dependencies.bash
COPY install_latest_cmake.bash /tmp/install_latest_cmake.bash

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        gnupg2 \
        lsb-release \
        build-essential \
        software-properties-common \
        cmake \
        git \
        tmux && \
    bash /tmp/install_latest_cmake.bash && \
    bash /tmp/install_onnx_runtime.bash && \
    bash /tmp/install_apps_dependencies.bash && \
    rm -rf /build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && rm /tmp/*


# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential software-properties-common \
        autoconf automake libtool pkg-config ca-certificates wget \
        git curl libjpeg-dev libpng-dev language-pack-en \
        locales locales-all python3 \
        python3-py python3-dev python3-pip python3-numpy \
        python3-pytest python3-setuptools libprotobuf-dev \
        protobuf-compiler zlib1g-dev swig vim gdb valgrind \
        libsm6 libxext6 libxrender-dev unzip sudo



# install Opencv
ENV OPENCV_VERSION=4.7.0
RUN cd /tmp \
    && wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
	&& unzip ${OPENCV_VERSION}.zip \
	&& rm ${OPENCV_VERSION}.zip
RUN cd /tmp/opencv-${OPENCV_VERSION} \
	&& mkdir build \
	&& cd build \
	&& cmake \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DENABLE_PRECOMPILED_HEADERS=OFF ..
RUN cd /tmp/opencv-${OPENCV_VERSION}/build \
    && make -j 8 \
	&& make install \
	&& ldconfig


# Install PyTorch for C++ https://pytorch.org
# LibTorch using cxx11 ABI instead of Pre-cxx11 ABI
ENV LIBTORCH_VERSION=2.5.1
ENV LIBTORCH_DOWNLOAD_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip

RUN wget ${LIBTORCH_DOWNLOAD_URL} -O libtorch.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip \
    && mv libtorch /usr/local/include/

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
