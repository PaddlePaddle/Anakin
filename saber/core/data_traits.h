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

#ifndef ANAKIN_SABER_CORE_DATA_TRAITS_H
#define ANAKIN_SABER_CORE_DATA_TRAITS_H

#include "saber_types.h"

#ifdef USE_BM
#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"
#endif

namespace anakin{

namespace saber{

template <typename Ttype>
struct DataTraitLp{
    typedef void* PtrDtype;
};

template <typename Ttype>
struct DataTraitBase{
    typedef void* PtrDtype;
};

#ifdef USE_OPENCL
template <>
struct DataTraitLp<AMD>{
    typedef cl_mem PtrDtype;
};

template <>
struct DataTraitBase<AMD>{
    typedef cl_mem PtrDtype;
};
#endif

static size_t type_length(DataType type) {
    switch (type){
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

template <typename Ttype, DataType datatype>
struct DataTrait{
    typedef __invalid_type Dtype;
    typedef __invalid_type PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_HALF> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_FLOAT> {
    typedef float Dtype;
    typedef float* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_DOUBLE> {
    typedef double Dtype;
    typedef double* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT8> {
    typedef char Dtype;
    typedef char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT16> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT32> {
    typedef int Dtype;
    typedef int* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT64> {
    typedef long Dtype;
    typedef long* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT8> {
    typedef unsigned char Dtype;
    typedef unsigned char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT16> {
    typedef unsigned short Dtype;
    typedef unsigned short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT32> {
    typedef unsigned int Dtype;
    typedef unsigned int* PtrDtype;
};

#ifdef USE_OPENCL
struct ClMem{
    ClMem(){
        dmem = nullptr;
        offset = 0;
    }

    ClMem(cl_mem mem_in, size_t offset_in = 0) {
        dmem = mem_in;
        offset = offset_in;
    }

    ClMem(const ClMem& right) {
        dmem = right.dmem;
        offset = right.offset;
    }

    ClMem& operator=(const ClMem& right) {
        this->dmem = right.dmem;
        this->offset = right.offset;
        return *this;
    }

    ClMem& operator+(const size_t offset_in) {
        this->offset += offset_in;
        return *this;
    }

    ClMem& operator ++() {
        this->offset += 1;
        return *this;
    }

    ClMem& operator ++(int) {
        this->offset += 1;
        return *this;
    }

    size_t offset{0};
    cl_mem dmem{nullptr};
};

template <>
struct DataTrait<AMD, AK_FLOAT> {
    typedef float Dtype;
    typedef cl_mem PtrDtype;
};

template <>
struct DataTrait<AMD, AK_DOUBLE> {
    typedef double Dtype;
    typedef cl_mem PtrDtype;
};

template <>
struct DataTrait<AMD, AK_INT8> {
    typedef char Dtype;
    typedef cl_mem PtrDtype;
};

template <>
struct DataTrait<AMD, AK_HALF> {
    typedef short Dtype;
    typedef cl_mem PtrDtype;
};
#endif //USE_OPENCL
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DATA_TRAITS_H
