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
#include <iostream>
#include <memory>
#include <vector>
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

#define CHECK_EQ(a, b) std::cout
#define CHECK_LE(a, b) std::cout
#define CHECK_LT(a, b) std::cout
#define CHECK_GE(a, b) std::cout
#define CHECK_GT(a, b) std::cout

#define LOG(a) std::cout

#define LITE_CHECK(condition) \
    do { \
    SaberStatus error = condition; \
    /*CHECK_EQ(error, SaberSuccess) << " " << get_error_string_lite(error);*/ \
} while (0)

inline const char* get_error_string_lite(SaberStatus error_code){
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
    }
    return "ANAKIN SABER UNKOWN ERRORS";
}

enum ARM_TYPE {
    ARM_CPU = 0,
    ARM_GPU = 1
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_COMMON_H

