FROM nvcr.io/nvidia/tensorrt:20.09-py3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata sudo

RUN apt-get update \
  && apt-get install -y ssh \
build-essential \
gcc \
g++ \
gdb \
clang \
make \
ninja-build \
cmake \
autoconf \
automake \
locales-all \
dos2unix \
rsync \
tar \
python \
python3-dev \
python3-pip \
ffmpeg libsm6 libxext6 \
bzip2 \
g++ \
git \
graphviz \
libgl1-mesa-glx \
libhdf5-dev \
openmpi-bin \
wget \
python3-tk && apt-get clean  &&  \
    rm -rf /var/lib/apt/lists/*

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && python3 get-pip.py --force-reinstall
RUN sudo pip3 install --no-cache-dir setuptools && pip3 install --no-cache-dir opencv-python==4.6.0.66 tqdm llvmlite numba scipy scikit-build numpy scipy imgaug scikit-image scikit-learn albumentations jupyterlab tqdm matplotlib pandas nanoid

# Minimize image size
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

RUN useradd -m user \
  && yes password | passwd user && adduser user sudo

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]