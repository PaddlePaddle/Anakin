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

template <DataType type>
struct DataTrait{
    typedef __invalid_type dtype;
};

template <>
struct DataTrait<AK_HALF> {
    typedef short dtype;
};

template <>
struct DataTrait<AK_FLOAT> {
    typedef float dtype;
};

template <>
struct DataTrait<AK_DOUBLE> {
    typedef double dtype;
};

template <>
struct DataTrait<AK_INT8> {
    typedef char dtype;
};

template <>
struct DataTrait<AK_INT16> {
    typedef short dtype;
};

template <>
struct DataTrait<AK_INT32> {
    typedef int dtype;
};

template <>
struct DataTrait<AK_INT64> {
    typedef long dtype;
};

template <>
struct DataTrait<AK_UINT8> {
    typedef unsigned char dtype;
};

template <>
struct DataTrait<AK_UINT16> {
    typedef unsigned short dtype;
};

template <>
struct DataTrait<AK_UINT32> {
    typedef unsigned int dtype;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DATA_TRAITS_H
