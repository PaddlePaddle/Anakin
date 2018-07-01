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

template <typename TargetType>
struct VoidPtr{
    VoidPtr(){
        offset = 0;
        ptr = nullptr;
    }
    VoidPtr(void* ptr_in, size_t offset_in = 0) {
        offset = offset_in;
        ptr = (char*)ptr_in + offset_in;
    }
    VoidPtr(const VoidPtr& right) {
        offset = right.offset;
        ptr = right.ptr;
    }
    VoidPtr&operator=(const VoidPtr& right) {
        this->offset = right.offset;
        this->ptr = right.ptr;
    }

    VoidPtr&operator+(const size_t offset_in) {
        this->offset += offset_in;
        ptr = (char*)ptr + offset_in;
        return *this;
    }

    size_t offset{0};
    void* ptr{nullptr};
};

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

template <>
struct DataTrait<AMD, AK_HALF> {
    typedef ClMem Dtype;
    typedef short dtype;
};

template <>
struct VoidPtr<AMD> {
    VoidPtr(){}

    ~VoidPtr() {
        if (!ptr) {
            delete ptr;
        }
    }

    VoidPtr(void* ptr_in, size_t offset_in = 0) {
        ptr = new ClMem;
        ClMem* mem_in = (ClMem*)ptr_in;
        ptr->dmem = mem_in->dmem;
        ptr->offset = mem_in->offset + offset_in;
    }
    VoidPtr(const VoidPtr& right) {
        ptr = new ClMem;
        ptr->dmem = right.ptr->dmem;
        ptr->offset = right.ptr->offset;
    }
    VoidPtr&operator=(const VoidPtr& right) {
        if (this->ptr == nullptr) {
            ptr = new ClMem;
        }
        this->ptr->dmem = right.ptr->dmem;
        this->ptr->offset = right.ptr->offset;
        return *this;
    }

    VoidPtr&operator+(const size_t offset_in) {
        this->ptr->offset += offset_in;
        return *this;
    }

    ClMem* ptr{nullptr};
};

#endif

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DATA_TRAITS_H
