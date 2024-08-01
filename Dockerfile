FROM nvcr.io/nvidia/tensorrt:22.12-py3


# Variables
ARG USERNAME=user
ARG UID=1000
ARG GID=1000
ARG WORKDIR=/workspace/BUSCA

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Paris"
ENV BUILDDIR=/workspace/build
ENV DATADIR=/beegfs/datasets


# User configuration
RUN groupadd -g ${GID} -o ${USERNAME}
RUN useradd -m -u ${UID} -g ${GID} -o -s /bin/bash ${USERNAME}
RUN echo "${USERNAME}:*" | chpasswd && adduser ${USERNAME} sudo
RUN passwd -d ${USERNAME}

RUN mkdir -p ${WORKDIR}
RUN chown -R ${USERNAME}:${USERNAME} ${WORKDIR}
RUN mkdir -p ${BUILDDIR}
RUN chown -R ${USERNAME}:${USERNAME} ${BUILDDIR}
RUN mkdir -p ${DATADIR}
RUN chown -R ${USERNAME}:${USERNAME} ${DATADIR}

WORKDIR ${BUILDDIR}


# General dependencies
RUN apt update
RUN apt install -y --no-install-recommends \
        automake autoconf libpng-dev nano python3-pip \
        curl ca-certificates zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc gcc-10 g++-10 cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev eog \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pip --upgrade
RUN pip3 install -U pip setuptools
RUN pip3 install --no-cache-dir gdown \
    && echo "export PATH=$PATH:/home/user/.local/bin" >> /home/user/.bashrc


# BUSCA dependencies
ADD requirements.txt ${BUILDDIR}/requirements.txt
RUN pip3 install --no-cache-dir -r ${BUILDDIR}/requirements.txt -f https://download.pytorch.org/whl/cu115

# To save on storage
RUN pip3 cache purge


# ByteTrack dependencies
RUN pip3 install --no-cache-dir \
    loguru \
    scikit-image \
    Pillow \
    thop \
    ninja \
    tabulate \
    tensorboard \
    lap \
    motmetrics \
    filterpy \
    h5py \
    onnx \
    onnxruntime \
    onnx-simplifier \
    pandas

RUN pip3 install --no-cache-dir cython \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 install cython_bbox

RUN ldconfig
RUN pip3 cache purge

RUN pip install setuptools==69.5.1  # To avoid "ImportError: cannot import name 'packaging' from 'pkg_resources'" when installing torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && git checkout 0400b38123d01cc845364870bdf0a0044ea2b3b2 \
    # https://github.com/NVIDIA-AI-IOT/torch2trt/issues/619
    && wget https://github.com/NVIDIA-AI-IOT/torch2trt/commit/8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && git apply 8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && python3 setup.py install


# TransCenter dependencies
# You need CUDA enabled during the Docker build (https://stackoverflow.com/a/61737404)
# MultiScaleDeformableAttention
RUN git clone https://gitlab.inria.fr/robotlearn/TransCenter_official \
    && cd TransCenter_official/to_install/ops/ \
    && python setup.py build install \
    && echo "/usr/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg" > /lib/python3/dist-packages/MultiScaleDeformableAttention.pth \
    || ( echo -e "ERROR: Could not install TransCenter dependencies! \n  This is probably due to CUDA not being available during the Docker build (https://stackoverflow.com/a/61737404)\n  Skipping TransCenter setup. Please, comment this line and rerun the Docker build with CUDA enabled to properly setup TransCenter or manually install its dependencies while running the container." && echo -e "echo -e '[WARNING] TransCenter dependencies were not properly installed during Docker build! \\\nIf you want to install them, run: \`cd /workspace/build/TransCenter_official/to_install/ops/ && sudo python setup.py build install && echo \"/usr/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg\" | sudo dd of=/lib/python3/dist-packages/MultiScaleDeformableAttention.pth && cd /workspace/BUSCA/\`'" >> /home/user/.bashrc && sleep 5 )

# dcn_v2
RUN cd TransCenter_official/to_install/DCNv2_torch1.11/ \
    && unzip DCNv2_torch1.11.zip \
    && python setup.py build develop \
    && echo "/workspace/build/TransCenter_official/to_install/DCNv2_torch1.11" > /lib/python3/dist-packages/DCNv2.pth


# GHOST dependencies
RUN pip3 install --no-cache-dir \
    lapsolver \
    torchreid


# CenterTrack dependencies
RUN pip3 install --no-cache-dir \
    progress \
    pyquaternion \
    numba

RUN pip3 cache purge


ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

USER ${USERNAME}
WORKDIR ${WORKDIR}
