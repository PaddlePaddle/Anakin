## Build Anakin for ARM from source ##

Now, we have successfully build on mac os and centos, using Android NDK

### Installation overview ###

* [system requirements](#0001)
* [dependencies](#0002)
* [build from source](#0003)
* [verification](#0004)


### <span id = '0001'> 1. system requirements </span> ###

*  Host machine: linux, mac    
*  cmake 3.8.2+    
*  Android NDK r14, download linux version from [here](https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip)

### <span id = '0002'> 2. dependencies </span> ###

- 2.1 protobuf3.4.0     
   Download source from https://github.com/google/protobuf/releases/tag/v3.4.0    
 - 2.1.1 Build protobuf for host     
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
    
 - 2.1.2 Build protobuf for ARM `armeabi-v7a`    
 ```bash
 
  ```
  Set your protobuf path [here](../../cmake/find_modules.cmake), search `anakin_find_protobuf`, and set `ARM_RPOTO_ROOT` to your path.    
  ```cmake
  set(ARM_RPOTO_ROOT "${CMAKE_SOURCE_DIR}/third-party/arm-android/protobuf")
  ```
  
- 2.2 opencv 2.4.3+(optional)    
    We only use opencv in examples   
    For Android, visit opencv [release page](https://opencv.org/releases.html), choose Android pack and download, 
    copy libs in `3rdparty/libs/armeabi-v7a` to `libs/armeabi-v7a`.   
    Set your opencv path [here](../../cmake/find_modules.cmake),  Search `anakin_find_opencv`, 
    and set `include_directories` and `LINK_DIRECTORIES` according to your path.   
    ```cmake
    include_directories(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/jni/include/)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/libs/armeabi-v7a/)
    ```
### <span id = '0003'> 3. build from source </span> ###

#### build for Android

   clone the [source code](https://github.com/PaddlePaddle/Anakin/tree/arm)
```bash
    cd your_dir
    git clone https://github.com/PaddlePaddle/Anakin.git
    cd Anakin
    git fetch origin arm
    git checkout arm
  ```
  change the `android_build.sh`    
- Set NDK path to yours    
  ```bash
    #modify "your_ndk_path" to your NDK path
    export ANDROID_NDK=your_ndk_path
  ```
- Set your ARM target platform    

  For 32bits ARM CPU with NEON, Set ANDROID_ABI to `armeabi-v7a with NEON`， 
  for 64bits ARM CPU, either `arm64-v8a` or `armeabi-v7a with NEON` can work.    
  Now, we only support `armeabi-v7a with NEON`，`arm64-v8a` is under developing    
  ```bash
      -DANDROID_ABI="armeabi-v7a with NEON"
  ```
- Set Android API level    
  Choose your API LEVEL according to your android system version    
  API Level 21 -> Android 5.0.1    
  ```bash
      -DANDROID_NATIVE_API_LEVEL=21
  ```

- build static or shared lib    
  if building static lib, set `BUILD_SHARED=NO`    
  if building shared lib, set `BUILD_SHARED=YES`    
  ```bash
      -DBUILD_SHARED=NO
  ```
- OpenMP for multi-threads    
  set `USE_OPENMP=YES` to use OpenMP multi-threads    
  ```bash
      -DUSE_OPENMP=YES
  ```
  
- build unit test    
  set `BUILD_WITH_UNIT_TEST=YES` to build unit tests    
    ```bash
        -DBUILD_WITH_UNIT_TEST=YES
    ```

- build examples    
  set `BUILD_EXAMPLES=YES` to build detection and classification examples    
    ```bash
        -DBUILD_EXAMPLES=YES
    ```
  
- use opencv in examples    
  set `USE_OPENCV=YES` to use opencv in examples    
    ```bash
        -DUSE_OPENCV=YES
    ```
    
- build    
  run `android_build.sh` to build the Anakin     
  ```bash
      ./android_build.sh
  ```

### <span id = '0004'> 4. Verification </span> ###    
  The libs is in `${Anakin_root}/output`, the unit test and benchmark file is in `${Anakin_root}/output/unit_test` 
  and the examples is in `${Anakin_root}/output/examples`   
  Open `USB debug mode` in your Android device, Use ADB to push the test files and model files to `data/local/tmp/your_dir`    
  run the test