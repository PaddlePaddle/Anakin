/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SABER_LITE_CORE_COMMON_H
#define ANAKIN_SABER_LITE_CORE_COMMON_H

//#include "utils/logger/logger.h"
#include <memory>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include "anakin_config.h"
#include "saber/saber_types.h"

#ifdef USE_ARM_PLACE
#include <arm_neon.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif //openmp
#endif //ARM

namespace anakin{

namespace saber{

namespace lite{

#if defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__
#  define LITE_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#  define LITE_EXPORT __attribute__ ((visibility ("default")))
#else
#  define LITE_EXPORT
#endif

//#define CHECK_EQ(a, b) std::cout
//#define CHECK_LE(a, b) std::cout
//#define CHECK_LT(a, b) std::cout
//#define CHECK_GE(a, b) std::cout
//#define CHECK_GT(a, b) std::cout
//#define LOG(a) std::cout

#define LCHECK_EQ(a, b, out) \
do { if (a != b) { printf("%s\n", out); assert(0);} } while (0)

#define LCHECK_GE(a, b, out) \
do { if (a < b) { printf("%s\n", out); assert(0);} } while (0)

#define LCHECK_GT(a, b, out) \
do { if (a <= b) { printf("%s\n", out); assert(0);} } while (0)

#define LCHECK_LE(a, b, out) \
do { if (a > b) { printf("%s\n", out); assert(0);} } while (0)

#define LCHECK_LT(a, b, out) \
do { if (a >= b) { printf("%s\n", out); assert(0);} } while (0)

#define LITE_CHECK(condition) \
    do { \
    SaberStatus error = condition; \
    if (error != SaberSuccess) { \
        printf("SaberLite runtime error type %s\n", get_error_string_lite(error)); \
        assert(0);\
    } \
} while (0)

inline const char* get_error_string_lite(SaberStatus error_code) {
    switch (error_code) {
        case SaberSuccess:
            return "ANAKIN_SABER_STATUS_SUCCESS";
        case SaberNotInitialized:
            return "ANAKIN_SABER_STATUS_NOT_INITIALIZED";
        case SaberInvalidValue:
            return "ANAKIN_SABER_STATUS_INVALID_VALUE";
        case SaberMemAllocFailed:
            return "ANAKIN_SABER_STATUS_MEMALLOC_FAILED";
        case SaberUnKownError:
            return "ANAKIN_SABER_STATUS_UNKNOWN_ERROR";
        case SaberOutOfAuthority:
            return "ANAKIN_SABER_STATUS_OUT_OF_AUTHORITH";
        case SaberOutOfMem:
            return "ANAKIN_SABER_STATUS_OUT_OF_MEMORY";
        case SaberUnImplError:
            return "ANAKIN_SABER_STATUS_UNIMPL_ERROR";
        case SaberWrongDevice:
            return "ANAKIN_SABER_STATUS_WRONG_DEVICE";
        default:
            return "ANAKIN SABER UNKOWN ERRORS";
    }
}
#if 0 //add support for opencl device memory
template <typename dtype>
struct CLDtype{
    CLDtype(){
        offset = 0;
        ptr = nullptr;
    }

    CLDtype& operator++(){
        offset++;
        return *this;
    }
    CLDtype operator++(int){

    }
    int offset;
    cl_mem ptr;
};
#endif

enum ARMType{
    CPU = 0,
    GPU = 1,
    DSP = 2
};

template <ARMType Ttype, DataType Dtype>
struct DataTrait{
    typedef void dtype;
};


template <ARMType Ttype>
struct DataTrait<Ttype, AK_FLOAT>{
    typedef float dtype;
    typedef float Dtype;
};

template <ARMType Ttype>
struct DataTrait<Ttype, AK_INT8>{
    typedef char dtype;
    typedef char Dtype;
};

template <ARMType Ttype>
struct TargetTrait{
    typedef void* stream_t;
    typedef void* event_t;
    typedef void bdtype;
    int get_device_count() { return 1;}
    int get_device_id(){ return 0;}
    void set_device_id(int id){}
};

//! the alignment of all the allocated buffers
const int MALLOC_ALIGN = 16;

static void* fast_malloc(size_t size) {
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p;
    p = static_cast<char*>(malloc(offset + size));
    if (!p) {
        return nullptr;
    }
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
}

static void fast_free(void* ptr) {
    if (ptr){
        free(static_cast<void**>(ptr)[-1]);
    }
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_COMMON_H

