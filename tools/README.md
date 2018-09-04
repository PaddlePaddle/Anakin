# Anakin Tools

## 1. Andrid Build Script

**[ Usage ]**
```bash
$ cd tools
$ ./andrid_build.sh  # it will generate directory 'android_build' in anakin root path.
```

**[ Build Option ]**

> 1. -DBUILD_SHARED=NO/OFF # you can build static and shared lib 
> 2. -DANDROID_ABI="armeabi-v7a with NEON" # api edition
> 3. -DANDROID_NDK # you need to use `export` to set env var ANDROID_NDK to the path of android ndk.


**[ Note ]**

1. if you choose shared libanakin.so, you need to push to the /path/to/lib that is writable, and then set LD_LIBRARAY_PATH to that path.
2. enjoy

## 2. IOS Build Script

> not support yet.

## 3. external_converter_v2

> Please refer to [this](external_converter_v2/README.md)

## 4. anakin-lite

> Please refer to [this](anakin-lite/README.md)
