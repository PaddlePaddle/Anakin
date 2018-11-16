/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
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

#include <memory>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <map>
#include <string.h>
#include <sstream>
#include "anakin_config.h"
#include "saber/saber_types.h"

#ifdef ENABLE_OP_TIMER
#include <sstream>
#endif

#ifdef USE_ARM_PLACE
#include <arm_neon.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif //openmp
#endif //ARM

namespace anakin{

namespace saber{

namespace lite{

#define MAJOR 1
#define MINOR 1
#define REVISION 0

#ifdef USE_ANDROID_LOG
#include "android/log.h"
#ifndef LOG_TAG
#define LOG_TAG "Anakin_lite"
#endif //log_tag
#endif // PLATFORM_ANDROID
#define LOG_NOOP (void) 0
//__FILE__ 输出文件名
//__LINE__ 输出行数
//__PRETTY_FUNCTION__  输出方法名
//可以按需选取 %s %u %s 分别与之对应
//通过IS_DEBUG来控制是否输出日志
#ifdef USE_ANDROID_LOG
#define LOG_PRINT(level, fmt, ...) \
	__android_log_print(level, LOG_TAG, "(%s:%u) %s: " fmt, \
	__FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOG_PRINT(ANDROID_LOG_INFO, fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__);
#define LOGD(fmt, ...) LOG_PRINT(ANDROID_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) LOG_PRINT(ANDROID_LOG_WARN, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOG_PRINT(ANDROID_LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOGF(fmt, ...) LOG_PRINT(ANDROID_LOG_FATAL, fmt, ##__VA_ARGS__)
#else
#define LOGI(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGD(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGF(fmt, ...) printf(fmt, ##__VA_ARGS__); assert(0)
#endif //USE_ANDROID_LOG

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

#define LCHECK_EQ(a, b, fmt,...) \
do { if (a != b) { LOGF(fmt,##__VA_ARGS__);} } while (0)

#define LCHECK_GE(a, b, fmt,...) \
do { if (a < b) { LOGF(fmt,##__VA_ARGS__);} } while (0)

#define LCHECK_GT(a, b, fmt,...) \
do { if (a <= b) { LOGF(fmt,##__VA_ARGS__);} } while (0)

#define LCHECK_LE(a, b, fmt,...) \
do { if (a > b) { LOGF(fmt,##__VA_ARGS__);} } while (0)

#define LCHECK_LT(a, b, fmt,...) \
do { if (a >= b) { LOGF(fmt,##__VA_ARGS__);} } while (0)

#define LITE_CHECK(condition) \
    do { \
    SaberStatus error = condition; \
    if (error != SaberSuccess) { \
        LOGF("SaberLite runtime error type %s\n", get_error_string_lite(error)); \
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

enum ARMArch{
    APPLE = 0,
    A53 = 53,
    A55 = 55,
    A57 = 57,
    A72 = 72,
    A73 = 73,
    A75 = 75,
    A76 = 76,
    ARM_UNKOWN = -1
};

template <ARMType Ttype>
struct DataTraitBase {
    typedef void* PtrDtype;
};

template <ARMType Ttype, DataType dtype>
struct DataTrait{
    typedef void Dtype;
};


template <ARMType Ttype>
struct DataTrait<Ttype, AK_FLOAT>{
    typedef float Dtype;
    typedef float dtype;
};

template <ARMType Ttype>
struct DataTrait<Ttype, AK_INT8>{
    typedef char Dtype;
    typedef char dtype;
};

template <ARMType Ttype>
struct TargetTrait{
    typedef void* stream_t;
    typedef void* event_t;
    typedef void* ptrtype;
    int get_device_count() { return 1;}
    int get_device_id(){ return 0;}
    void set_device_id(int id){}
};

static size_t type_length(DataType type) {
    switch (type) {
        case AK_INT8:
            return 1;
        case AK_UINT8:
            return 1;
        case AK_INT16:
            return 2;
        case AK_UINT16:
            return 2;
        case AK_INT32:
            return 4;
        case AK_UINT32:
            return 4;
        case AK_INT64:
            return 8;
        case AK_HALF:
            return 2;
        case AK_FLOAT:
            return 4;
        case AK_DOUBLE:
            return 8;
        default:
            return 4;
    }
}

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

static void* align_ptr(void* ptr, size_t offset) {
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(static_cast<char*>(ptr) + offset) & (~(MALLOC_ALIGN - 1)));
    return r;
}

#if 1//def ENABLE_OP_TIMER

struct GOPS{
    float ts;
    float ops;
    GOPS operator+(const GOPS& right) {
        GOPS out;
        out.ts = this->ts + right.ts;
        out.ops = this->ops + right.ops;
        return out;
    }
};

class OpTimer {
public:
    static std::map<std::string, GOPS>& ops() {
        static std::map<std::string, GOPS>* _timer = new std::map<std::string, GOPS>();
        return *_timer;
    }
    // Adds a timer type.
    static void add_timer(const std::string& type, GOPS ts) {
        std::map<std::string, GOPS>& _timer = ops();
        if (_timer.count(type) < 1) {
            _timer[type] = ts;
        } else {
            GOPS tn = _timer[type] + ts;
            _timer[type] = tn;
        }
    }

    static GOPS get_timer(const std::string type) {
        std::map<std::string, GOPS>& _timer = ops();
        if (_timer.count(type) < 1) {
            LOGE("unknow type: %s\n", type.c_str());
            return {0.f, 0.f};
        }
        return _timer[type];
    }

    static void print_timer() {
        std::map<std::string, GOPS>& _timer = ops();
        GOPS to = get_timer("total");
        if (to.ts <= 0.f) {
            to.ts = 1.f;
        }
        for (auto& it : _timer) {
            printf("op: %s, timer: %f, GOPS: %f, percent: %f%%\n", \
                it.first.c_str(), it.second.ts, 1e-6f * it.second.ops / it.second.ts, 100.f * it.second.ts / to.ts);
        }
    }

private:
    OpTimer() {}
};

#endif //ENABLE_TIMER

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_COMMON_H

