# Base image with CUDA 12.8 and cuDNN
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"

# ---- Base packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    gdb \
    git \
    less \
    htop \
    wget \
    cmake\
    python3-dev \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    libarchive-tools \
    libboost-all-dev \
    clang-format \
    ffmpeg \
    libzip-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install -U pip && \
    pip install \
        gdown \
        numpy \
        pandas \
        matplotlib \
        jupyter \
        cmake \
        --break-system-packages && \
    python3 -m pip cache purge && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Install CMake 3.25.2 ----
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh && \
    chmod +x cmake-3.25.2-linux-x86_64.sh && \
    ./cmake-3.25.2-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.25.2-linux-x86_64.sh

# ---- Install PyTorch + LibTorch with CUDA 12.8 ----
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    wget -qO- https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip | bsdtar -xvf- -C /usr/local

# ---- Install Arcade Learning Environment (ALE C++ version) ----
RUN cd /opt && \
    git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git ale && \
    cd ale && \
    git checkout v0.8.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DUSE_SDL=OFF -DUSE_RLGL=OFF .. && \
    make -j$(nproc) && make install && \
    rm -rf /opt/ale

# ---- Install pybind11 v2.4.3 ----
RUN git clone https://github.com/pybind/pybind11.git /opt/pybind11 && \
    cd /opt/pybind11 && \
    git checkout v2.4.3 && \
    mkdir build && cd build && \
    cmake .. \
      -DPYBIND11_TEST=OFF \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make install -j$(nproc) && \
    rm -rf /opt/pybind11

# ---- Environment variables ----
ENV CMAKE_PREFIX_PATH="/usr/local/libtorch/share/cmake/Torch:/usr/local"
ENV LD_LIBRARY_PATH="/usr/local/libtorch/lib:$LD_LIBRARY_PATH"

# ---- Create working directory ----
RUN mkdir /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

# Default command
CMD ["/bin/bash"]
