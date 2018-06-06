# Tools
---

## Tools list
---
> andrid_build.sh
> ios_build.sh
> gpu_build.sh

## 1. Andrid Build Script
---

**[ Usage ]**
```bash
$ cd tools
$ ./andrid_build.sh  # it will generate directory 'android_build' in anakin root path.
```

**[ Build Option ]**
> -DBUILD_SHARED=NO/OFF # you can build static and shared lib 
> -DANDROID_ABI="armeabi-v7a with NEON" # api edition
> -DANDROID_NDK # you need to use `export` to set env var ANDROID_NDK to the path of android ndk.

**[ Note ]**
1. if you choose shared libanakin.so, you need to push to the /path/to/lib that is writable, and then set LD_LIBRARAY_PATH to that path.
2. enjoy

## 2. IOS Build Script
---
not impl yet.
