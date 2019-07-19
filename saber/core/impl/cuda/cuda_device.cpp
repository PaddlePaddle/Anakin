/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "core/device.h"
namespace anakin{

namespace saber{

#ifdef USE_CUDA
template <>
void Device<NV>::create_stream() {
	_data_stream.clear();
	_compute_stream.clear();
	for(int i = 0; i < _max_stream; i++) {
		typedef TargetWrapper<NV> API;
		typename API::stream_t stream_data;
		typename API::stream_t stream_compute;
		//cudaStreamNonBlocking
		API::create_stream_with_flag(&stream_data, 1);
		API::create_stream_with_flag(&stream_compute, 1);
		_data_stream.push_back(stream_data);
		_compute_stream.push_back(stream_compute);
	}
}

template <>
void Device<NV>::get_info() {
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	_info._idx = dev;
	_info._device_name = std::string(deviceProp.name);
	LOG(INFO) << "Device id: " << dev << " , name: " << deviceProp.name;
	_info._compute_core_num = deviceProp.multiProcessorCount;
	LOG(INFO) << "Multiprocessors: " << deviceProp.multiProcessorCount;
	_info._max_frequence = deviceProp.clockRate / 1000;
	_info._min_frequence = deviceProp.clockRate / 1000;
	LOG(INFO) << "frequency:" << deviceProp.clockRate / 1000 << "MHz";
	_info._generate_arch = deviceProp.major * 10 + deviceProp.minor;
	LOG(INFO) << "CUDA Capability : " << deviceProp.major << "." << deviceProp.minor;
	_info._compute_ability = std::to_string(_info._generate_arch);
	_info._max_memory = deviceProp.totalGlobalMem / 1048576;
	LOG(INFO) << "total global memory: " << deviceProp.totalGlobalMem / 1048576 << "MBytes.";
};

template <>
void Device<NVHX86>::create_stream() {
    //todo
    //LOG(ERROR) << "NVHX86 create_stream is not implemented";
	_data_stream.clear();
	_compute_stream.clear();
	for(int i = 0; i < _max_stream; i++) {
		typedef TargetWrapper<NVHX86> API;
		typename API::stream_t stream_data;
		typename API::stream_t stream_compute;
		//cudaStreamNonBlocking
		API::create_stream_with_flag(&stream_data, 1);
		API::create_stream_with_flag(&stream_compute, 1);
		_data_stream.push_back(stream_data);
		_compute_stream.push_back(stream_compute);
	}
}

template <>
void Device<NVHX86>::get_info() {
    // todo
    LOG(ERROR) << "NVHX86 get_info is not implemented";
};

template void Device<NV>::create_stream();
template void Device<NV>::get_info();

template void Device<NVHX86>::create_stream();
template void Device<NVHX86>::get_info();

#endif //USE_CUDA

} //namespace saber
} //namespace anakin

