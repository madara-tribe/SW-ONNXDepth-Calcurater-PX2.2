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

WORKDIR /workspace
RUN apt-get update && apt-get install -y vim
ENTRYPOINT ["/bin/bash"]
