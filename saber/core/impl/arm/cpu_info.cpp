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

#include <cstdarg>
#include "saber/core/impl/arm/cpu_info.h"
namespace anakin{

namespace saber{

#ifdef PLATFORM_ANDROID

// cache_id : 0 -> L1, 1 -> L2, 2 -> L3
void set_cache_info(DeviceInfo<ARM>& cpu_info, int cache_id, int argc, ...){
    va_list arg_ptr;
    va_start(arg_ptr, argc);
    std::vector<int>* cache;
    switch (cache_id){
        case 0:
            cache = &cpu_info._L1_cache;
            break;
        case 1:
            cache = &cpu_info._L2_cache;
            break;
        case 2:
            cache = &cpu_info._L3_cache;
            break;
        default:
            break;
    }
    int core_num = cpu_info._compute_core_num;
    cache->resize(core_num);
    if (argc == 1){
        int cache_size = va_arg(arg_ptr, int);
        for (int i = 0; i < core_num; ++i){
            (*cache)[i] = cache_size;
        }
    } else {
        int big_core_num = cpu_info._big_core_ids.size();
        int little_core_num = cpu_info._little_core_ids.size();
        int big_core_cache_size = va_arg(arg_ptr, int);
        int little_core_cache_size = va_arg(arg_ptr, int);
        for (int i = 0; i < big_core_num; ++i){
            (*cache)[cpu_info._big_core_ids[i]] = big_core_cache_size;
        }
        for (int i = 0; i < little_core_num; ++i){
            (*cache)[cpu_info._little_core_ids[i]] = little_core_cache_size;
        }
    }
    va_end(arg_ptr);
}

void set_arch_info(DeviceInfo<ARM>& cpu_info, int argc, ...){
    va_list arg_ptr;
    va_start(arg_ptr, argc);
    int core_num = cpu_info._compute_core_num;
    cpu_info._archs.resize(core_num);
    if (argc == 1){
        ARMArch arch = (ARMArch)va_arg(arg_ptr, int);
        for (int i = 0; i < core_num; ++i){
            cpu_info._archs[i] = arch;
        }
    } else {
        ARMArch big_core_arch = (ARMArch)va_arg(arg_ptr, int);
        ARMArch little_core_arch = (ARMArch)va_arg(arg_ptr, int);
        int big_core_num = cpu_info._big_core_ids.size();
        int little_core_num = cpu_info._little_core_ids.size();
        for (int i = 0; i < big_core_num; ++i){
            cpu_info._archs[cpu_info._big_core_ids[i]] = big_core_arch;
        }
        for (int i = 0; i < little_core_num; ++i){
            cpu_info._archs[cpu_info._little_core_ids[i]] = little_core_arch;
        }
    }
    va_end(arg_ptr);
}

SaberStatus get_cpu_info_from_name(DeviceInfo<ARM>& cpu_info, std::string hardware_name){

    /* Snapdragon */

    if (hardware_name.find("SM8150") != std::string::npos){ //855
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A76, A55);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 256 * 1024, 128 * 1024);
        set_cache_info(cpu_info, 2, 1, 2048 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("SDM845") != std::string::npos){ //845
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A75, A55);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 256 * 1024, 128 * 1024);
        set_cache_info(cpu_info, 2, 1, 2048 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("SDM710") != std::string::npos){ //710
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3, 4, 5};
        cpu_info._cluster_ids = {1, 1, 1, 1, 1, 1, 0, 0};
        set_arch_info(cpu_info, 2, A75, A55);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 256 * 1024, 128 * 1024);
        set_cache_info(cpu_info, 2, 1, 1024 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MSM8998") != std::string::npos){ //835
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A73, A53);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 1024 * 1024,
                /*real cache size is 2M, while that will get bad performace on conv3x3s1 or gemm, set to 1M or 512K*/
                       1024 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MSM8996") != std::string::npos){ //820
        cpu_info._compute_core_num = 4;
        cpu_info._core_ids = {0, 1, 2, 3};
        cpu_info._big_core_ids = {2, 3};
        cpu_info._little_core_ids = {0, 1};
        cpu_info._cluster_ids = {1, 1, 0, 0};
        set_arch_info(cpu_info, 1, A72);
        set_cache_info(cpu_info, 0, 1, 24 * 1024);
        set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("SDM660") != std::string::npos ||
               hardware_name.find("SDM636") != std::string::npos){ // 660, 636
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A73, A53);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 1, 1024 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MSM8976") != std::string::npos){ // 652,653
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A72, A53);
        set_cache_info(cpu_info, 0, 1, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MSM8953") != std::string::npos){ // 625
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._little_core_ids = {};
        cpu_info._cluster_ids = {0, 0, 0, 0, 0, 0, 0, 0};
        set_arch_info(cpu_info, 1, A53);
        set_cache_info(cpu_info, 0, 1, 32 * 1024);
        set_cache_info(cpu_info, 1, 1, 1024 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MSM8939") != std::string::npos){ // 615
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {0, 1, 2, 3};
        cpu_info._little_core_ids = {4, 5, 6, 7};
        cpu_info._cluster_ids = {0, 0, 0, 0, 1, 1, 1, 1};
        set_arch_info(cpu_info, 1, A53);
        set_cache_info(cpu_info, 0, 1, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 512 * 1024, 256 * 1024);
        return SaberSuccess;

    /* MediaTek */

    } else if (hardware_name.find("MT6797") != std::string::npos){ // X20/X23/X25/X27
        cpu_info._compute_core_num = 10;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        cpu_info._big_core_ids = {8, 9};
        cpu_info._little_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._cluster_ids = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
        set_arch_info(cpu_info, 2, A72, A53);
        set_cache_info(cpu_info, 0, 1, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
        return SaberSuccess;

    } else if (hardware_name.find("MT6799") != std::string::npos){ // X30
        cpu_info._compute_core_num = 10;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        cpu_info._big_core_ids = {8, 9};
        cpu_info._little_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._cluster_ids = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
        set_arch_info(cpu_info, 2, A73, A53);
        return SaberSuccess;

    }else if (hardware_name.find("MT6795")  != std::string::npos ||
              hardware_name.find("MT6762")  != std::string::npos ||
              hardware_name.find("MT6755T") != std::string::npos ||
              hardware_name.find("MT6755S") != std::string::npos ||
              hardware_name.find("MT6753")  != std::string::npos ||
              hardware_name.find("MT6752")  != std::string::npos ||
              hardware_name.find("MT6750")  != std::string::npos){ // X10, P22, P15/P18, MT6753 \
                                                                      MT6752/MT6752M, MT6750
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._little_core_ids = {};
        cpu_info._cluster_ids = {0, 0, 0, 0, 0, 0, 0, 0};
        set_arch_info(cpu_info, 1, A53);
        return SaberSuccess;

    } else if (hardware_name.find("MT6758")  != std::string::npos ||
               hardware_name.find("MT6757")  != std::string::npos ||
               hardware_name.find("MT6763")  != std::string::npos ||
               hardware_name.find("MT6755M") != std::string::npos ||
               hardware_name.find("MT6755")  != std::string::npos){ // P30, P20/P25, P23, P10
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 1, A53);
        return SaberSuccess;

    } else if (hardware_name.find("MT6771")  != std::string::npos){ // P60
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A73, A53);
        return SaberSuccess;

    } else if (hardware_name.find("MT6765") != std::string::npos ||
               hardware_name.find("MT6739") != std::string::npos ||
               hardware_name.find("MT6738") != std::string::npos ||
               hardware_name.find("MT6737") != std::string::npos){ // A22, MT6739, MT6738, MT6767
        cpu_info._compute_core_num = 4;
        cpu_info._core_ids = {0, 1, 2, 3};
        cpu_info._big_core_ids = {0, 0, 0, 0};
        cpu_info._little_core_ids = {};
        cpu_info._cluster_ids = {0, 0, 0, 0};
        set_arch_info(cpu_info, 1, A53);
        return SaberSuccess;

        /* Kirin */
    } else if (hardware_name.find("KIRIN980") != std::string::npos){ // Kirin 980
        cpu_info._compute_core_num = 8;
        cpu_info._core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
        cpu_info._big_core_ids = {4, 5, 6, 7};
        cpu_info._little_core_ids = {0, 1, 2, 3};
        cpu_info._cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
        set_arch_info(cpu_info, 2, A76, A55);
        set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
        set_cache_info(cpu_info, 1, 2, 512 * 1024, 128 * 1024);
        set_cache_info(cpu_info, 2, 1, 4096 * 1024);
        return SaberSuccess;
    }

    return SaberUnImplError;
}

#endif


} //namespace saber

} //namespace anakin
