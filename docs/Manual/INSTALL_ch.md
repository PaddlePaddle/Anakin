# 从源码编译安装Anakin #

我们已经在CentOS 7.3，Ubuntu 16.04 上安装和测试了Anakin。

# 安装概览 #

[TOC]


## 在CentOS上安装 Anakin
## 1. 系统要求

*  make 3.82+
*  cmake 2.8.12+
*  gcc 4.8.2+
*  g++ 4.8.2+
*  其他需要补充的。。。

## 2. 编译和安装依赖
###  2.1.1 protobuf
    >$ git clone https://github.com/google/protobuf
    >$ cd protobuf
    >$ git submodule update --init --recursive
    >$ ./autogen.sh
    >$ ./configure --prefix=/path/to/your/insall_dir
    >$ make
    >$ make check
    >$ make install
    >$ sudo ldconfig
    如安装protobuf遇到任何问题，请访问[这里](https://github.com/google/protobuf/blob/master/src/README.md)

### 2.2 CUDA Toolkit(不使用NV的GPU可以跳过这一步)
  - [CUDA 8.0](https://developer.nvidia.com/cuda-zone) or higher. 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
  - [cuDNN v7](https://developer.nvidia.com/cudnn). 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
  为了帮助cmake找到对应的cudnn，请手动导出CUDNN_ROOT变量如下
```bash
    export CUDNN_ROOT=/usr/local/cudnn_v7 # the path where your cuDNN installed.
```


## 3. 编译Anakin

### 3.1 克隆Anakin源码
>$ git clone https://github.com/PaddlePaddle/Anakin.git

### 3.2.0 手动编译编译Anakin
修改cmake编译选项中DUSE_X86_PLACE为false可以关闭X86的编译，其他选项参考[根目录的CMakeList](CMakeList.txt)
>$ make build
>$ cd build
>$ cmake .. -DUSE_GPU_PLACE=true -DUSE_X86_PLACE=true
### 3.2.1 使用脚本编译Anakin(手动编译，脚本编译二选一即可)
>$ bash tools/nv_gpu_build.sh
>$ bash tools/x86_build.sh

## 在Ubuntu上安装 Anakin

已支持Ubuntu 16.04 上安装Anakin，步骤与CentOS类似

## 在ARM上安装 Anakin

请参考[ARM安装文档](run_on_arm_ch.md)
Lite安装请参考[ARM-LITE安装文档](run_on_arm_lite_ch.md)

## 安装支持AMD的Anakin
请参考[英文版安装文档](INSTALL_en.md)

## 验证安装

安装完成后，如果没有报错信息，你可以通过运行 `output/unit_test`路径下的单测示例验证是否编译成功。



