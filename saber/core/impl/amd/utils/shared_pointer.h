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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_SHAREDPOINTER_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_SHAREDPOINTER_H
#include <memory>
#include <iostream>
#include "anakin_config.h"
#include "amd_logger.h"

namespace anakin {
namespace saber {

template <class F, F f>
struct SharedDeleter {
    template <class T>
    void operator()(T* t) const {
        if (t != NULL) {
            try {
                AMD_LOGD("release shared object :" << typeid(t).name());
                f(t);
                t = NULL;
            } catch (...) {
                AMD_LOGE("Catught error for release shared object : " << typeid(t).name());
            }
        }
    }
};

#define SHARED_OBJ(T) std::shared_ptr<std::remove_pointer<T>::type>

#define GEN_SHARED_OBJ_WITH_DELETER(T, F, t)       \
    std::shared_ptr<std::remove_pointer<T>::type>( \
            t, anakin::saber::SharedDeleter<decltype(&F), &F>())

#define GEN_SHARED_OBJ(T, t) std::shared_ptr<std::remove_pointer<T>::type>(t)

} // namespace saber
} // namespace anakin

#endif
