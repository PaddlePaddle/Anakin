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
#ifndef ANAKIN_SABER_CORE_IMPL_AMD_UTILS_AMDLOGGER_H
#define ANAKIN_SABER_CORE_IMPL_AMD_UTILS_AMDLOGGER_H

#include "anakin_config.h"
#include "utils/logger/logger.h"

namespace anakin {
namespace saber {
//#define AMD_ENABLE_LOG

#define _AMD_LOGD(X) std::cout << X << std::endl;
#define _AMD_LOGE(X) std::cerr << X << std::endl;

#define AMD_LOGI(X) LOG(INFO) << X
#define AMD_LOGE(X) LOG(ERROR) << X

#define ALOGI(X) LOG(INFO) << X
#define ALOGE(X) LOG(ERROR) << X

#if defined(ENABLE_DEBUG) || defined(AMD_ENABLE_LOG)
#define AMD_LOGD(X) LOG(INFO) << X
#define ALOGD(X) LOG(INFO) << X
#else
#define AMD_LOGD(X)
#define ALOGD(X)
#endif

} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_CORE_IMPL_AMD_UTILS_AMDLOGGER_H
