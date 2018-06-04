################################################################################
#   Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
################################################################################

FROM centos:7.4.1708

# anakin install ubuntu GPU env
RUN yum -y install vim wget git make glibc-devel libstdc++-deve epel-release gcc gcc-c++ libstdc++ && rm -rf /var/cache/yum/*

RUN yum -y install python-pip && rm -rf /var/cache/yum/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask numpy pyyaml scipy pandas

# set env
ENV LIBRARY_PATH /usr/lib64:$LIBRARY_PATH

# install cmake
RUN wget https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz && tar xzf cmake-3.2.0.tar.gz && \
    cd cmake-3.2.0 && ./bootstrap && \
    make -j4 && make install && cd .. && rm -f cmake-3.2.0.tar.gz

# install protobuf
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
    && tar -xvf protobuf-cpp-3.4.0.tar.gz \
    && cd protobuf-3.4.0 && ./configure \
    && make -j4 && make install && cd .. \
    && rm -f protobuf-cpp-3.4.0.tar.gz

RUN echo "[ROCm]" > /etc/yum.repos.d/rocm.repo \
    && echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo \
    && echo "baseurl=http://repo.radeon.com/rocm/yum/rpm" >> /etc/yum.repos.d/rocm.repo \
    && echo "enabled=1" >> /etc/yum.repos.d/rocm.repo \
    && echo "gpgcheck=0" >> /etc/yum.repos.d/rocm.repo

RUN yum -y install rocm-opencl rocm-opencl-devel && rm -rf /var/cache/yum/*

# set env
ENV LIBRARY_PATH /opt/rocm/lib:/opt/rocm/opencl/lib/x86_64:$LIBRARY_PATH
ENV OCL_ROOT /opt/rocm/opencl/lib/x86_64
ENV PATH /opt/rocm/bin:/opt/rocm/opencl/bin/x86_64:$PATH

# build and install anakin
#RUN git clone --branch master --recursive http://gitlab.baidu.com/cuichaowen/anakin2.git \
#    && cd anakin2/tools && ./gpu_build.sh
