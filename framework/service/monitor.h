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

#include "framework/service/device_info.h"

namespace anakin {

namespace rpc {

/// monitor thread pool
template<typename Ttype>
class Monitor {
public:
    Monitor(){}
    ~Monitor(){}

    template<Info ...infos>
    void create_instance(int dev_id, int interval_time_in_sec) {
        _id = dev_id;
        _monitor_thread = new std::thread([this](int dev_id, int time) {
            DevInfo<infos...> dev_info_pack;
            std::chrono::time_point<sys_clock> start = sys_clock::now(); 
            for(;;) {
                double elapsed_time_ms =\
                        std::chrono::duration_cast<std::chrono::milliseconds>(sys_clock::now()-start).count();
                if(elapsed_time_ms  > time * 1000) {
                    dev_info_pack.template inquiry<Ttype>(dev_id);
                    _name = dev_info_pack.template get<DEV_NAME>();
                    _temp = dev_info_pack.template get<DEV_TMP>();
                    _mem_free = dev_info_pack.template get<DEV_MEM_FREE>();
                    _mem_used = dev_info_pack.template get<DEV_MEM_USED>();
                    LOG(INFO) << "Device ("<<dev_id << ")";
                    LOG(INFO) << "|-- Temp  : \t"<<_temp;
                    LOG(INFO) << "|-- Free  : \t"<<_mem_free;
                    LOG(INFO) << "`-- Used  : \t"<<_mem_used;

                    start = sys_clock::now();
                }
            }
        }, dev_id, interval_time_in_sec);
    }

    int get_id() { return _id; } 

    std::string get_name() { return _name; }

    float get_temp() { return _temp; }

    float get_mem_free() { return _mem_free; }

    float get_mem_used() { return _mem_used; }

private:
    typedef std::chrono::system_clock sys_clock;
    int _id{-1};   // device id (represent as device num id)
    std::string _name{"unknown"};     // device name
    float _temp{-1000};     // device temperature Celsius degree
    float _mem_free{-1}; // device memory free bytes
    float _mem_used{-1};
    std::thread* _monitor_thread;
}; 

} /* namespace rpc */

} /* namespace anakin */

#endif
