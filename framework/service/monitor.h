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

#ifndef ANAKIN_MONITOR_H
#define ANAKIN_MONITOR_H 

#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>
#include <functional>
#include <mutex> 
#include <chrono>

#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types

namespace anakin {

namespace rpc {

/*struct DevInfo{
    DevInfo() {}
    // all of those funcs must be thread safety
    void fill_id(int id) {
        std::unique_lock<std::mutex> lock(this->mut);
        dev_id = id;    
    }
    void set_temp(int);
    void set_g_mem_free(size_t);
    void set_g_mem_used(size_t);
    void set_name(const char*);
    void set_compute_run_proc_num(int);

    // 
    int get_id() { return dev_id; }
    int get_temp() { return temp; }
    size_t get_g_mem_free() { return g_mem_free; }
    size_t get_g_mem_used() { return g_mem_used; }
    std::string get_name() { return dev_name; }
    int get_compute_run_proc() { return compute_run_proc_num; }

private:
    // resource infomation
    int              dev_id{-1}; // device id current reside
    int                temp{-1}; // device temperature  
    size_t       g_mem_free{-1}; // global memory free (bytes)
    size_t       g_mem_used{-1}; // global memory used (bytes)
    int compute_run_proc_num{0}; // compute running process num on device
    std::string        dev_name; // device name
    std::mutex mut;
};


class Monitor {
}; */

} /* namespace rpc */

} /* namespace anakin */

#endif
