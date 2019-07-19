#include "framework/service/device_info.h"

namespace anakin {

namespace rpc {

#ifdef USE_CUDA
template<>
typename InfoTraits<DEV_ID>::data_type Inquiry<saber::NV>::get<DEV_ID>() {
    return _dev_id;
}

template<>
typename InfoTraits<DEV_NAME>::data_type Inquiry<saber::NV>::get<DEV_NAME>() {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE]; 
    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE); 
    if (NVML_SUCCESS != result) { 
        LOG(FATAL) << "Failed to get name of device: " << nvmlErrorString(result); 
    }
    return std::string(name);
}

template<>
typename InfoTraits<DEV_TMP>::data_type Inquiry<saber::NV>::get<DEV_TMP>() { 
    nvmlTemperatureSensors_t sensorType = NVML_TEMPERATURE_GPU;
    unsigned int temp; 
    result = nvmlDeviceGetTemperature(device, sensorType, &temp); 
    if (NVML_SUCCESS != result) { 
        LOG(FATAL) << "Failed to get temperature of device: " << nvmlErrorString(result); 
    }
    return temp;
}

template<>
typename InfoTraits<DEV_MEM_FREE>::data_type Inquiry<saber::NV>::get<DEV_MEM_FREE>() { 
    if(!memory_has_inspected) {
        result = nvmlDeviceGetMemoryInfo(device, &memory); 
        if (NVML_SUCCESS != result) { 
            LOG(FATAL) << "Failed to get device memory info of device: " << nvmlErrorString(result); 
        }
        memory_has_inspected = true;
    }
    return memory.free;
}

template<>
typename InfoTraits<DEV_MEM_USED>::data_type Inquiry<saber::NV>::get<DEV_MEM_USED>() { 
    if(!memory_has_inspected) {
        result = nvmlDeviceGetMemoryInfo(device, &memory); 
        if (NVML_SUCCESS != result) { 
            LOG(FATAL) << "Failed to get device memory info of device: " << nvmlErrorString(result); 
        } 
        memory_has_inspected = true;
    }
    return memory.used; 
}
#endif

} /* namespace rpc */

} /* namespace anakin */

