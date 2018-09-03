FROM centos:7.4.1708

# anakin install basic tools
RUN yum -y install zip unzip vim wget git make glibc-devel libstdc++-devel epel-release gcc gcc-c++ libstdc++ && rm -rf /var/cache/yum/*

# uncomment if needed.
#RUN yum -y install python-pip && rm -rf /var/cache/yum/*
#RUN pip install --no-cache-dir --upgrade pip && \
#    pip install --no-cache-dir \
#        flask numpy pyyaml scipy pandas


# set env
#ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# install cmake
RUN wget https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz && tar xzf cmake-3.2.0.tar.gz && \
        cd cmake-3.2.0 && ./bootstrap && \
        make -j4 && make install && cd .. && rm -f cmake-3.2.0.tar.gz

# install pre-requires before installing protobuf.
RUN yum -y install autoconf automake libtool

# install protobuf on Host.
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
    && tar -xvf protobuf-cpp-3.4.0.tar.gz \
    && cd protobuf-3.4.0 && ./configure \
    && make -j4 && make install && make distclean && cd .. \
    && rm -f protobuf-cpp-3.4.0.tar.gz

# Clone Anakin source code(arm branch).
RUN git clone --branch arm_lite --recursive "https://github.com/PaddlePaddle/Anakin.git"

# Download Android NDK r14 (Linux version)
# uncompress it and produce android-ndk-r14b directory.
RUN wget https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
RUN unzip -q android-ndk-r14b-linux-x86_64.zip && rm -f android-ndk-r14b-linux-x86_64.zip

RUN cd Anakin/tools/ \
    && ./build_lite.sh

# set cross-compile environment.
ENV ANDROID_NDK /android-ndk-r14b
RUN cd Anakin/tools/anakin-lite/ \
    && sed -i "s/export ANDROID_NDK=\/Users\/lixiaoyang05\/android-ndk-r14b/#/g" lite_android_build_armv7.sh \
    && ./lite_android_build_armv7.sh