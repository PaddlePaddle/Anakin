
FROM centos:7.4.1708

# anakin install ubuntu GPU env
RUN yum -y install vim wget git make glibc-devel libstdc++-deve epel-release gcc gcc-c++ libstdc++ openssl-devel \
    && rm -rf /var/cache/yum/*

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

RUN yum -y install rocm-opencl rocm-opencl-devel miopengemm && rm -rf /var/cache/yum/*

# set env
ENV LIBRARY_PATH /opt/rocm/lib:/opt/rocm/opencl/lib/x86_64:$LIBRARY_PATH
ENV OCL_ROOT /opt/rocm/opencl/lib/x86_64
ENV PATH /opt/rocm/bin:/opt/rocm/opencl/bin/x86_64:$PATH

RUN git clone --branch dev_v2 --recursive "https://github.com/PaddlePaddle/Anakin.git" /root/Anakin && cd /root/Anakin/tools/ && ./amd_gpu_build.sh && cd -

