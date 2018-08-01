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

#include "core/env.h"
namespace anakin{

namespace saber{

#ifdef AMD_GPU


typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

cl_platform_id AMD_ENV::platform_id = NULL;

void AMD_ENV::env_init(int max_stream){
    Devs& devs = cur_env();
    if (devs.size() > 0){
        return;
    }

    platform_id = AMD_API::get_platform_id();

    int count = 0;
    AMD_API::get_device_count(count);
    if (count == 0) {
        LOG(WARNING) << "no device found!";
    } else {
        LOG(INFO) << "found " << count << " device(s)";
    }

    int cur_id = AMD_API::get_device_id();
    for (int i = 0; i < count; i++) {
        AMD_API::set_device(i);
        devs.push_back(Device<AMD>(max_stream));
    }
    AMD_API::set_device(cur_id);
}

bool AMD_ENV::is_init(){
    CHECK(platform_id != NULL);
    return true;
}

cl_platform_id AMD_ENV::get_platform_id(){
    if(!is_init()) {
        return NULL;
    }
    return platform_id;
}



//template void AMD_ENV::evn_init();

#endif // AMD_GPU

} //namespace saber
} //namespace anakin

