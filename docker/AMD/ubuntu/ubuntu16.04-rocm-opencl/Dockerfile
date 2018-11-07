
FROM ubuntu:16.04

# anakin install ubuntu GPU env
RUN apt-get update
RUN apt-get -y install cmake make build-essential git libssl-dev wget libnuma-dev

# set env
ENV LIBRARY_PATH /usr/lib64:$LIBRARY_PATH

# install protobuf
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
    && tar -xvf protobuf-cpp-3.4.0.tar.gz \
    && cd protobuf-3.4.0 && ./configure \
    && make -j4 && make install && cd .. \
    && rm -f protobuf-cpp-3.4.0.tar.gz

RUN wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - \
    && echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | tee /etc/apt/sources.list.d/rocm.list

RUN apt-get update \
    && apt-get -y install rocm-dev rocm-cmake miopengemm

# set env
ENV LIBRARY_PATH /opt/rocm/lib:/opt/rocm/opencl/lib/x86_64:$LIBRARY_PATH
ENV OCL_ROOT /opt/rocm/opencl/lib/x86_64
ENV PATH /opt/rocm/bin:/opt/rocm/opencl/bin/x86_64:$PATH

RUN git clone --branch dev_v2 --recursive "https://github.com/PaddlePaddle/Anakin.git" /root/Anakin && cd /root/Anakin/tools/ && ./amd_gpu_build.sh && cd -
