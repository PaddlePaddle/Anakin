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

#ifndef ANAKIN_SABER_LITE_CORE_CPU_INFO_H
#define ANAKIN_SABER_LITE_CORE_CPU_INFO_H

#include "saber/core/device.h"
namespace anakin{

namespace saber{

#ifdef PLATFORM_ANDROID

SaberStatus get_cpu_info_from_name(DeviceInfo<ARM>& cpu_info, std::string hardware_name);

#endif

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_CPU_INFO_H
