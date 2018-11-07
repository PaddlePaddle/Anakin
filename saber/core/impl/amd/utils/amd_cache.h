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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_AMDCACHE_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_AMDCACHE_H
#include <memory>
#include <map>
#include <iostream>
#include "ocl/ocl_kernel.h"
#include "anakin_config.h"

namespace anakin {
namespace saber {

template <typename T, typename U>
class AMDCache {
public:
    typedef std::map<U, T> MType;

    static AMDCache<T, U>* getInstance() {
        if (_instance.get() == NULL) {
            _instance = std::unique_ptr<AMDCache<T, U>>(new AMDCache<T, U>);
        }
        return _instance.get();
    }

    size_t getSize() {
        return cache_size;
    }

    // lookup the binary given the file name
    T lookup(U key) {
        typename MType::iterator iter;
        iter = cache_map.find(key);
        if (iter != cache_map.end()) {
            AMD_LOGD("[" << typeid(T).name() << "] found cache!");
            return iter->second;
        } else
            return NULL;
    }

    // add program to the cache
    void add(U key, T t) {

        if (!lookup(key)) {
            AMD_LOGD("[" << typeid(T).name() << "] insert cache to slot, size: " << cache_size);
            cache_size++;
            cache_map.insert(typename MType::value_type(key, t));
            AMD_LOGD("[" << typeid(T).name() << "] insert cache to slot done");
        }
    }

    // The presumed watermark for the cache volume (256MB). Is it enough?
    // We may need more delicate algorithms when necessary later.
    // Right now, let's just leave it along.
    static const unsigned MAX_CACHE_SIZE = 1024;

    ~AMDCache() {
        release();
    }

    AMDCache() {
        cache_size = 0;
        cache_map.clear();
    };

protected:
    void release() {
        cache_size = 0;
        cache_map.clear();
    }

private:
    static typename std::unique_ptr<AMDCache<T, U>> _instance;
    MType cache_map;
    unsigned int cache_size;
};

template <class T, typename U>
std::unique_ptr<AMDCache<T, U>> AMDCache<T, U>::_instance = NULL;

using ProgramCache = AMDCache<ClProgramPtr, std::pair<cl_context, std::string>>;
using KernelCache  = AMDCache<ClKernelPtr, std::pair<cl_program, std::string>>;

} // namespace saber
} // namespace anakin

#endif
