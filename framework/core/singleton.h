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

#ifndef ANAKIN_SINGLETON_H
#define ANAKIN_SINGLETON_H 

#include "anakin_config.h"
#include "framework/core/thread_safe_macros.h"
#include <mutex>
#ifdef USE_SGX
#include <support/sgx/sgx_mutex>
#endif

namespace anakin {

typedef void(*ReleaseAtExit)();

/// Default exit function. do nothing.
void default_exit();

template<typename T, ReleaseAtExit release_func = default_exit>
class Singleton {
public:
    static T& Global() EXCLUSIVE_LOCKS_REQUIRED(_sg_mutex) {
        if (!_instance) {
            std::lock_guard<std::mutex> guard(_sg_mutex);
            if (!_instance) {
                _instance = new T();
                atexit(release_func);
            }
        }
        return *_instance;
    }

private:
    Singleton();
    ~Singleton();

    //! disable copy construction
    Singleton(const Singleton&);
    //! disable copy
    Singleton& operator=(const Singleton&);

    static T* _instance GUARDED_BY(_sg_mutex);
    static std::mutex _sg_mutex;
};

template<typename T, ReleaseAtExit release_func> 
T* Singleton<T, release_func>::_instance = nullptr;

template<typename T, ReleaseAtExit release_func> 
std::mutex Singleton<T, release_func>::_sg_mutex;


} /* namespace anakin */

#endif
