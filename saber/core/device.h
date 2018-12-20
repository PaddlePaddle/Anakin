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

#ifndef ANAKIN_SABER_CORE_DEVICE_H
#define ANAKIN_SABER_CORE_DEVICE_H
#include "core/target_wrapper.h"
#include <string>

namespace anakin {

namespace saber {

template <typename TargetType>
struct DeviceInfo {
    int _idx;
    std::string _device_name;
    int _max_frequence;
    int _min_frequence;
    std::string _compute_ability;
    int _generate_arch;
    int _compute_core_num;
    int _max_memory;
    int _sharemem_size;
    int _L1_cache;
    int _L2_cache;
    int _L3_cache;
    std::vector<int> _core_ids;
    std::vector<int> _cluster_ids;
};

template <typename TargetType>
struct Device {

    typedef TargetWrapper<TargetType> API;

    Device(int max_stream = 4) : _max_stream(max_stream) {
        get_info();
//        create_stream();
    }
    void get_info();
    void create_stream();
    DeviceInfo<TargetType> _info;
    int _max_stream;

    std::vector<typename API::stream_t> _data_stream;
    std::vector<typename API::stream_t> _compute_stream;
};

#ifdef AMD_GPU
template <>
struct Device<AMD> {

    typedef TargetWrapper<AMD> API;

    Device(int max_stream = 1);

    void get_info();
    void create_stream();
    DeviceInfo<AMD> _info;
    int _max_stream;

    std::vector<typename API::stream_t> _data_stream;
    std::vector<typename API::stream_t> _compute_stream;

    cl_device_id get_device() {
        return id;
    };
    cl_context get_context() {
        return context;
    };

    typename API::stream_t get_available_stream(typename API::stream_t default_stream = nullptr);

private:
    cl_device_id id;
    cl_context context;
};


template struct Device<AMDHX86>;

#endif
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DEVICE_H
