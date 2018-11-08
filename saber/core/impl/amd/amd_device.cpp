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

#include "core/device.h"
#include "core/env.h"

namespace anakin {

namespace saber {

#ifdef AMD_GPU

typedef TargetWrapper<AMD> AMD_API;
typedef TargetWrapper<AMDHX86> AMDHX86_API;

size_t split(const std::string& txt, std::vector<std::string>& strs, char ch) {
    size_t pos = txt.find(ch);
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while (pos != std::string::npos) {
        strs.push_back(txt.substr(initialPos, pos - initialPos));
        initialPos = pos + 1;

        pos = txt.find(ch, initialPos);
    }

    // Add the last one
    strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));
    return strs.size();
}

template <typename T>
static void get_param(cl_device_id dev, cl_device_info param_name, T** param_value) {
    size_t valueSize;
    clGetDeviceInfo(dev, param_name, 0, NULL, &valueSize);
    T* value = (T*)malloc(valueSize);
    clGetDeviceInfo(dev, param_name, valueSize, value, NULL);
    *param_value = value;
}


Device<AMD>::Device(int max_stream) : _max_stream(max_stream) {
    if (!Env<AMD>::is_init()) {
        return;
    }

    //get cl device id;
    int nums = 0;
    AMD_API::get_device_count(nums);
    cl_device_id* device_ids = new cl_device_id[nums];
    cl_uint device_nums;
    clGetDeviceIDs(Env<AMD>::get_platform_id(), CL_DEVICE_TYPE_GPU, (cl_uint)nums, device_ids,
                   &device_nums);
    id = device_ids[AMD_API::get_device_id()];
    delete[] device_ids;

    //init context, one by one mapping to device.
    cl_int errNum;
    const cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)Env<AMD>::get_platform_id(), 0};
    context = clCreateContext(prop, 1, &id, NULL, NULL, &errNum);
    CHECK(errNum == CL_SUCCESS);

    get_info();
    create_stream();
}

void Device<AMD>::create_stream() {
    _data_stream.clear();
    _compute_stream.clear();

    for (int i = 0; i < _max_stream; i++) {
        typename AMD_API::stream_t stream_data;
        typename AMD_API::stream_t stream_compute;

#ifdef ENABLE_AMD_PROFILING
        API::_create_stream_with_flag(&stream_data, context, id, CL_QUEUE_PROFILING_ENABLE);
        API::_create_stream_with_flag(&stream_compute, context, id, CL_QUEUE_PROFILING_ENABLE);
#else
        API::_create_stream_with_flag(&stream_data, context, id, 0);
        API::_create_stream_with_flag(&stream_compute, context, id, 0);
#endif
        _data_stream.push_back(stream_data);
        _compute_stream.push_back(stream_compute);
    }
}

void Device<AMD>::get_info() {

    _info._idx = AMD_API::get_device_id();

    char* name;
    get_param(id, CL_DEVICE_NAME, &name);
    _info._device_name = std::string(name);
    free(name);

    cl_uint* num;
    get_param(id, CL_DEVICE_MAX_COMPUTE_UNITS, &num);
    _info._compute_core_num = *num;
    free(num);

    get_param(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, &num);
    _info._max_frequence = *num;
    _info._min_frequence = *num;
    free(num);

    get_param(id, CL_DEVICE_VERSION, &name);
    std::string version = std::string(name);
    std::vector<std::string> strs;
    split(version, strs, ' ');
    _info._generate_arch = (int)(stof(strs[1]) * 10);
    free(name);

    cl_ulong* size;
    get_param(id, CL_DEVICE_GLOBAL_MEM_SIZE, &size);
    _info._max_memory = (int)(*size / 1048576);
    free(size);

    LOG(INFO) << "Device id: " << _info._idx << " , name: " << _info._device_name;
    LOG(INFO) << "Multiprocessors: " << _info._compute_core_num;
    LOG(INFO) << "frequency:" << _info._max_frequence << " MHz";
    LOG(INFO) << "AMD OpenCL Capability : " << _info._generate_arch;
    LOG(INFO) << "total global memory: " << _info._max_memory << " MBytes.";
};

typename AMD_API::stream_t Device<AMD>::get_available_stream(typename AMD_API::stream_t stream) {
    if (stream == nullptr) {
        return _data_stream[0];
    }

    cl_device_id t_device_id;

    if (clGetCommandQueueInfo(stream, CL_QUEUE_DEVICE, sizeof(cl_device_id), &t_device_id,
                              NULL) == CL_SUCCESS) {
        if (t_device_id == id) {
            return stream;
        }

    }

    LOG(INFO) << "Can't find this stream use default data stream to instead";
    return _data_stream[0];

}

template <>
void Device<AMDHX86>::create_stream() {
    _data_stream.clear();
    _compute_stream.clear();

    for (int i = 0; i < _max_stream; i++) {
        typedef TargetWrapper<AMDHX86> API;
        typename API::stream_t stream_data;
        typename API::stream_t stream_compute;

        API::create_stream_with_flag(&stream_data, 1);
        API::create_stream_with_flag(&stream_compute, 1);
        _data_stream.push_back(stream_data);
        _compute_stream.push_back(stream_compute);
    }
}

template <>
void Device<AMDHX86>::get_info() {
    LOG(ERROR) << "AMDHX86 get_info is not implemented";
};


#endif // AMD_GPU

} //namespace saber
} //namespace anakin

