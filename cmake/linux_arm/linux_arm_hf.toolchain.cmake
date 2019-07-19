# this one is important
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)
#this one not so much
#SET(CMAKE_SYSTEM_VERSION 1)

# specify the cross compiler
SET(CMAKE_C_COMPILER   ${LINUX_ARM_TOOL_ROOT}/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER ${LINUX_ARM_TOOL_ROOT}/bin/arm-linux-gnueabihf-g++)
#SET(CMAKE_LINKER /home/xuhailong/dev-tool/arm-linux/64hf/bin/arm-linux-gnueabihf-g++)
#SET(CMAKE_AR /home/xuhailong/dev-tool/arm-linux/64hf/bin/arm-linux-gnueabihf-g++)

# where is the target environment 
SET(CMAKE_FIND_ROOT_PATH  ${LINUX_ARM_TOOL_ROOT})

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# float-abi: hard, softfp
add_compile_options(-mfloat-abi=hard)
add_compile_options(-mfpu=neon)
add_compile_options(-march=armv7-a)
