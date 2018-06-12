## 从源码编译安装Anakin ##

我们目前支持Android平台，采用NDK交叉编译工具链
已在mac os和centos上编译通过

### 安装概览 ###

* [系统需求](#0001)
* [安装依赖](#0002)
* [Anakin 源码编译](#0003)
* [验证安装](#0004)


### <span id = '0001'> 1. 系统要求 </span> ###

*  宿主机 linux, mac
*  cmake 3.8.2+
*  Android NDK r14, download linux version from [here](https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip)

### <span id = '0002'> 2. 安装依赖 </span> ###

- 2.1 protobuf3.4.0

   Download source from https://github.com/google/protobuf/releases/tag/v3.4.0
 - 2.1.1 编译宿主机  
 ```bash
   $ tar -xzf protobuf-3.4.0.tar.gz  
   $ cd protobuf-3.4.0   
   $ ./autogen.sh  
   $ ./configure    
   $ make  
   $ make check   
   $ make install
   ```
   for details, please refer [here](https://github.com/google/protobuf/blob/v3.4.0/src/README.md)
    
 - 2.1.2 交叉编译 `armeabi-v7a`
 ```bash
 
  ```
    
- 2.2 opencv 2.4.3+(optional)
    
    visit opencv [release page](https://opencv.org/releases.html), choose Android pack and download

### <span id = '0003'> 3. Anakin 源码编译 </span> ###

#### build for Android

   从github下载[源码](https://github.com/PaddlePaddle/Anakin/tree/arm)
```bash
    cd your_dir
    git clone https://github.com/PaddlePaddle/Anakin/tree/arm
    cd Anakin
    git fetch origin arm
    git checkout arm
  ```
  修改tools目录下的`android_build.sh`
- 配置NDK路径
  ```bash
    #modify "your_ndk_path" to your NDK path
    export ANDROID_NDK=your_ndk_path
  ```
- 配置cpu架构

  设置ANDROID_ABI，32位或64位处理器可以设置为 `armeabi-v7a with NEON`， 64位处理器可以设置为`arm64-v8a`
  目前只支持编译`armeabi-v7a with NEON`，`arm64-v8a`正在开发中
  
  ```bash
      -DANDROID_ABI="armeabi-v7a with NEON"
  ```
- 配置Android API level

  根据发布的android系统版本选择API LEVEL
  ```bash
      -DANDROID_NATIVE_API_LEVEL=21
  ```

- 选择编译静态库或动态库

  设置`BUILD_SHARED=NO`编译静态库，设置`BUILD_SHARED=YES`编译动态库
  ```bash
      -DBUILD_SHARED=NO
  ```
- OpenMP 多线程
  设置`USE_OPENMP=YES`开启多线程
  ```bash
      -DUSE_OPENMP=YES
  ```
- 编译
  运行`android_build.sh`编译
  ```bash
      ./android_build.sh
  ```

### <span id = '0004'> 4. 验证安装 </span> ###

  在`output/unit_test`中编译出operator的单测文件和`benchmark_arm`模型性能测试文件，
  打开ANdroid设备的usb调试模式，通过android ADB工具将测试文件和模型文件推送到Android系统
  `data/local/tmp/your_dir`目录下运行



