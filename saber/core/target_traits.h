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

#ifndef ANAKIN_SABER_CORE_TARGET_TRAITS_H
#define ANAKIN_SABER_CORE_TARGET_TRAITS_H
#include "core/common.h"

namespace anakin{

namespace saber{

struct __host_target{};
struct __device_target{};

struct __cuda_device{};
struct __arm_device{};
struct __amd_device{};
struct __x86_device{};
struct __bm_device{};

struct __HtoD{};
struct __HtoH{};
struct __DtoD{};
struct __DtoH{};


template <class TargetType>
struct TargetTypeTraits {
    typedef __invalid_type target_category;
    typedef __invalid_type target_type;
};

template <>
struct TargetTypeTraits<NVHX86> {
    typedef __host_target target_category;
    typedef __x86_device target_type;
};
template <>
struct TargetTypeTraits<NV> {
    typedef __device_target target_category;
    typedef __cuda_device target_type;
};

template <>
struct TargetTypeTraits<X86> {
    typedef __host_target target_category;
    typedef __x86_device target_type;
};

template <>
struct TargetTypeTraits<ARM> {
    typedef __host_target target_category;
    typedef __arm_device target_type;
};

template <>
struct TargetTypeTraits<AMD> {
  typedef __device_target target_category;
  typedef __amd_device target_type;
};

template <>
struct TargetTypeTraits<BM> {
  typedef __device_target target_category;
  typedef __bm_device target_type;
};

template <>
struct TargetTypeTraits<AMDHX86> {
  typedef __host_target target_category;
  typedef __x86_device target_type;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_TARGET_TRAITS_H
