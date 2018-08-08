/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_MEM_INFO_H
#define ANAKIN_MEM_INFO_H 

#include "framework/core/parameter.h"
#include "framework/core/singleton.h"

namespace anakin {

/** 
 *  \brief memory management
 */
template<typename Ttype>
class MemInfo {
public:
	MemInfo() {}
	~MemInfo() {}

	/// get used mem in MB
	double get_used_mem_in_mb() { 
		return mem_used;
	}

private:
	double mem_used{0.f}; ///< mem in mb
	double mem_total{0.f}; //< mem in mb
};

#ifdef USE_CUDA
template<>
double MemInfo<NV>::get_used_mem_in_mb() {
	size_t free_bytes;
	size_t total_bytes;
	auto cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);	
	if(cudaSuccess != cuda_status) {
		LOG(FATAL) <<" cudaMemGetInfo fails: %s" << cudaGetErrorString(cuda_status);
	}
	this->mem_used = (double)(total_bytes - free_bytes)/1e6;
	this->mem_total = (double)total_bytes/1e6;
	return this->mem_used;
};
#endif

template<typename Ttype>
using MemoryInfo= Singleton<MemInfo<Ttype>>;

} /* namespace anakin */

#endif
