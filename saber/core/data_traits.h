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

namespace anakin{

namespace saber{

template <typename Ttype, DataType datatype>
struct DataTrait{
    typedef __invalid_type Dtype;
    typedef __invalid_type dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_HALF> {
    typedef short Dtype;
    typedef short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_FLOAT> {
    typedef float Dtype;
    typedef float dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_DOUBLE> {
    typedef double Dtype;
    typedef double dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT8> {
    typedef char Dtype;
    typedef char dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT16> {
    typedef short Dtype;
    typedef short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT32> {
    typedef int Dtype;
    typedef int dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT64> {
    typedef long Dtype;
    typedef long dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT8> {
    typedef unsigned char Dtype;
    typedef unsigned char dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT16> {
    typedef unsigned short Dtype;
    typedef unsigned short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT32> {
    typedef unsigned int Dtype;
    typedef unsigned int dtype;
};

#ifdef USE_OPENCL
struct ClMem{
    ClMem(){
        dmem = nullptr;
        offset = 0;
    }

    ClMem(cl_mem* mem_in, int offset_in = 0) {
        dmem = mem_in;
        offset = offset_in;
    }

    ClMem(ClMem& right) {
        dmem = right.dmem;
        offset = right.offset;
    }

    ClMem& operator=(ClMem& right) {
        this->dmem = right.dmem;
        this->offset = right.offset;
        return *this;
    }

    ClMem& operator+(int offset_in) {
        this->offset += offset_in;
        return *this;
    }

    int offset{0};
    cl_mem* dmem{nullptr};
};

template <>
struct DataTrait<AMD, AK_FLOAT> {
    typedef ClMem Dtype;
    typedef float dtype;
};

template <>
struct DataTrait<AMD, AK_DOUBLE> {
    typedef ClMem Dtype;
    typedef double dtype;
};

template <>
struct DataTrait<AMD, AK_INT8> {
    typedef ClMem Dtype;
    typedef char dtype;
};

#endif

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DATA_TRAITS_H
