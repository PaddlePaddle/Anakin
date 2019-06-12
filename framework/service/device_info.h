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

#ifndef ANAKIN_DEVICE_INFO_H
#define ANAKIN_DEVICE_INFO_H 

#include "anakin_config.h"

#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>
#include <functional>
#include <mutex> 
#include <chrono>

#ifdef USE_CUDA
#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types
#endif

#include "utils/logger/logger.h"
#include "saber/saber_types.h"

namespace anakin {

namespace rpc {

enum Info {
    DEV_ID,
    DEV_NAME,
    DEV_TMP,
    DEV_MEM_FREE,
    DEV_MEM_USED,
};

template<Info I1, Info I2>
struct check_same {
    const static bool value = false;
};

template<>
struct check_same<DEV_ID, DEV_ID> {
    const static bool value = true;
};

template<>
struct check_same<DEV_NAME, DEV_NAME> {
    const static bool value = true;
};

template<>
struct check_same<DEV_TMP, DEV_TMP> {
    const static bool value = true;
};

template<>
struct check_same<DEV_MEM_FREE, DEV_MEM_FREE> {
    const static bool value = true;
};

template<>
struct check_same<DEV_MEM_USED, DEV_MEM_USED> {
    const static bool value = true;
};


template<Info I>
struct InfoTraits {
    typedef float data_type;
};

template<>
struct InfoTraits<DEV_NAME> {
    typedef std::string data_type;
};

template<>
struct InfoTraits<DEV_ID> {
    typedef int data_type;
};

template<Info I>
struct InfoStruct {
    void _set(typename InfoTraits<I>::data_type value) {
        _val = value;
    }
    typename InfoTraits<I>::data_type _get() {
        return _val;
    }
private:
    typename InfoTraits<I>::data_type _val;
};

template<typename Ttype>
struct Inquiry;

#ifdef USE_CUDA
template<>
struct Inquiry<saber::NV> {
    ~Inquiry() {
        result = nvmlShutdown(); 
        if (NVML_SUCCESS != result) { 
            LOG(FATAL) << "Failed to shutdown the nvml of device: " << nvmlErrorString(result); 
        }
    }

    void init(int dev_id = 0) {
        _dev_id = dev_id;
        memory_has_inspected = false;
        result = nvmlInit(); 
        if (NVML_SUCCESS != result) { 
            LOG(FATAL) <<" Failed to initialize NVML: " << nvmlErrorString(result); 
        }
        result = nvmlDeviceGetHandleByIndex(dev_id, &device);
        if (NVML_SUCCESS != result) { 
            LOG(FATAL) << " Failed to get handle for device: " << nvmlErrorString(result); 
        }
    }

    template<Info I>
    typename InfoTraits<I>::data_type get();

private:
    int _dev_id;
    nvmlReturn_t result; 
    nvmlDevice_t device;
    nvmlMemory_t memory;
    bool memory_has_inspected;
};
#endif

template<Info target, Info info, Info ...infos>
struct HasTarget {
    const static bool value = check_same<target, info>::value || HasTarget<target, infos...>::value;
};

template<Info target, Info info>
struct HasTarget<target, info> { 
    const static bool value = check_same<target, info>::value; 
};

template<Info ...infos>
class DevInfo : public InfoStruct<infos>... {
public:
    template<Info I>
    void set(typename InfoTraits<I>::data_type value) {
        std::unique_lock<std::mutex> lock(this->_mut);
        if (!HasTarget<I, infos...>::value) {
            LOG(FATAL)<<" DevInfo parameter pack doesn't have target info type " << I;
        }
        InfoStruct<I>::_set(value);
    }

    template<Info I>
    typename InfoTraits<I>::data_type get() { 
        if (!HasTarget<I, infos...>::value) { 
            LOG(WARNING)<<" DevInfo parameter pack doesn't have target info type " << I; 
            return typename InfoTraits<I>::data_type();
        }
        return InfoStruct<I>::_get();
    }

    template<typename Ttype>
    void inquiry(int dev_id) {
        Inquiry<Ttype> instance;
        instance.init(dev_id);
        std::vector<Info> info_vec = {infos...};
        for (auto& info : info_vec) {
            switch(info) {
            case DEV_ID: {
                set<DEV_ID>(instance.get<DEV_ID>());
            } break;
            case DEV_NAME: {
                set<DEV_NAME>(instance.get<DEV_NAME>());
            } break;
            case DEV_TMP: {
                set<DEV_TMP>(instance.get<DEV_TMP>());
            } break;
            case DEV_MEM_FREE: {
                set<DEV_MEM_FREE>(instance.get<DEV_MEM_FREE>());
            } break;
            case DEV_MEM_USED: {
                set<DEV_MEM_USED>(instance.get<DEV_MEM_USED>());
            } break;
            default: break;
            }
        }
    }
private:
    std::mutex _mut;
};

} /* namespace rpc */

} /* namespace anakin */

#endif
