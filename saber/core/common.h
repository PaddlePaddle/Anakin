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

#ifndef ANAKIN_SABER_CORE_COMMON_H
#define ANAKIN_SABER_CORE_COMMON_H

#include <iostream>
#include <vector>
#include <type_traits>
#include <typeinfo>
#include <stdlib.h>
#include <map>
#include <list>

#include "utils/logger/logger.h"
#include "anakin_config.h"
#include "saber/saber_types.h"

namespace anakin{

namespace saber{

#define SABER_CHECK(condition) \
    do { \
    SaberStatus error = condition; \
    CHECK_EQ(error, SaberSuccess) << " " << saber_get_error_string(error); \
} while (0)

inline const char* saber_get_error_string(SaberStatus error_code){
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

template <bool If, typename ThenType, typename ElseType>
struct IF {
    /// Conditional type result
    typedef ThenType Type;      // true
};

template <typename ThenType, typename ElseType>
struct IF<false, ThenType, ElseType> {
    typedef ElseType Type;      // false
};

} //namespace saber

} //namespace anakin

#ifdef USE_CUDA
//#include <cuda.h>
#include <cuda_runtime.h>

const int CUDA_NUM_THREADS = 512;

#define CUDA_KERNEL_LE(i, n) \
  int i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < n)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); i += blockDim.x * gridDim.x)

/// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#endif // USE_CUDA

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << cublas_get_errorstring(status); \
  } while (0)
const char* cublas_get_errorstring(cublasStatus_t error);
#endif //USE_CUBLAS

#ifdef USE_CURAND
#include <curand.h>
#endif //USE_CURAND

#ifdef USE_CUFFT
#include <cufft.h>
#endif //USE_CUFFT

#ifdef USE_CUDNN
#include <cudnn.h>
#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnn_get_errorstring(status); \
  } while (0)

const char* cudnn_get_errorstring(cudnnStatus_t status);
#endif //USE_CUDNN

#ifdef USE_AMD

#ifdef __APPLE__
#include <OpenCL/cl_ext.h>
#include <OpenCL/cl.h>
#else
#include <CL/cl_ext.h>
#include <CL/cl.h>
#endif

#define AMD_CHECK_MSG(condition, msg) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cl_int error = condition; \
    CHECK_EQ(error, CL_SUCCESS) << " " << msg << " (err=" << opencl_get_error_string(error) << ")"; \
  } while (0)


#define AMD_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cl_int error = condition; \
    CHECK_EQ(error, CL_SUCCESS) << " " <<  opencl_get_error_string(error); \
  } while (0)
#endif


#ifdef USE_ARM_PLACE
#ifdef USE_OPENMP
#include <omp.h>
#include <arm_neon.h>
#endif //openmp
#endif //ARM

#endif //ANAKIN_SABER_CORE_COMMON_H

#ifdef USE_BM

#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmdnn_ext_api.h"
#include "bmlib_utils.h"

#define BMDNN_CHECK(condition) \
  do { \
    bm_status_t error = condition; \
    CHECK_EQ(error, BM_SUCCESS) << " Failed with error code:" << error; \
  } while (0)

#endif // USE_BM

