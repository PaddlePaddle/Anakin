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

   Part of the following code in this file refs to https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp

   Tencent is pleased to support the open source community by making ncnn available.

   Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.

   Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
   in compliance with the License. You may obtain a copy of the License at

   https://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed
   under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.
*/

#include "device.h"
#include "context.h"

#ifdef USE_ARM_PLACE

#ifdef PLATFORM_ANDROID
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/system_properties.h>
#include "cpu_info.h"
#endif //PLATFORM_ANDROID

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#endif //TARGET_OS_IPHONE
#endif //__APPLE__

namespace anakin{

namespace saber{

int arm_get_cpucount() {
#ifdef PLATFORM_ANDROID
    // get cpu count from /sys/devices/system/cpu/cpunum/uevent
    int max_cpu_count = 20;
    int count = 0;
    for (int i = 0; i < max_cpu_count; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", i);
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            break;
        }
        count++;
        fclose(fp);
    }
    if (count < 1) {
        count = 1;
    }
    return count;
#elif defined(TARGET_IOS)
    int count = 0;
    size_t len = sizeof(count);
    sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
    if (count < 1) {
        count = 1;
    }
    return count;
#else
    return 1;
#endif
}

void arm_get_cpu_arch(std::vector<ARMArch>& archs){
#ifdef PLATFORM_ANDROID
    archs.clear();
    //! get CPU ARCH
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return;
    }
    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s) {
            break;
        }
        if (strstr(line, "part") != NULL) {
            int arch_id = 0;
            sscanf(s, "CPU part\t: %x", &arch_id);
            switch (arch_id) {
                case 0xd03:
                    archs.push_back(A53);
                    break;
                case 0xd05:
                    archs.push_back(A55);
                    break;
                case 0xd07:
                    archs.push_back(A57);
                    break;
                case 0xd08:
                    archs.push_back(A72);
                    break;
                case 0xd09:
                    archs.push_back(A73);
                    break;
                case 0xd0a:
                    archs.push_back(A75);
                    break;
                case 0xd40:
                    archs.push_back(A76);
                    break;
                case 0x804:
                    // 855
                    archs.push_back(A76);
                    break;
                case 0x805:
                    // 855
                    archs.push_back(A55);
                    break;
                case 0x802:
                    // 845
                    archs.push_back(A75);
                    break;
                case 0x803:
                    // 845
                    archs.push_back(A55);
                    break;
                case 0x800:
                    // 835
                    archs.push_back(A73);
                    break;
                case 0x205:
                    // 820
                    archs.push_back(A72);
                    break;
                default:
                    LOG(ERROR) << "unknow type";
                    archs.push_back(ARM_UNKOWN);
            }
        }
    }
    fclose(fp);
    int cpu_count = arm_get_cpucount();
    if (archs.size() < cpu_count) {
        for (int i = archs.size(); i < cpu_count; ++i) {
            archs.push_back(archs[i - 1]);
        }
    }
#endif
#ifdef TARGET_IOS
    int cpu_count = arm_get_cpucount();
    for(int i = 0; i < cpu_count; ++i){
        archs.push_back(APPLE);
    }
#endif
}

void set_default_cache(DeviceInfo<ARM>& dev){
    int cpu_count = arm_get_cpucount();
    dev._L1_cache.resize(cpu_count);
    dev._L2_cache.resize(cpu_count);
    dev._L3_cache.resize(cpu_count);
#ifdef TARGET_IOS
    for (int i = 0; i < cpu_count; ++i){
        dev._L1_cache[i] = 64 * 1024;
        dev._L2_cache[i] = 2048 * 1024;
        dev._L3_cache[i] = 0;
    }
#else
    for (int i = 0; i < cpu_count; ++i){
        dev._L1_cache[i] = 32 * 1024;
        dev._L2_cache[i] = 512 * 1024;
        dev._L3_cache[i] = 0;
    }
#endif
}

size_t arm_get_meminfo() {
#ifdef PLATFORM_ANDROID
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/meminfo", "rb");
    if (!fp) {
        return 1;
    }

    size_t memsize = 0;
    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s) {
            break;
        }
        sscanf(s, "MemTotal:        %d kB", &memsize);
    }

    fclose(fp);

    return memsize;
#elif defined(TARGET_IOS)
    // to be implemented
    printf("not implemented\n");
    return 0;
#endif
}

#ifdef PLATFORM_ANDROID
std::string arm_get_cpu_name(){
    std::string cpu_name;
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return "";
    }
    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s) {
            break;
        }
        if (strstr(line, "Hardware") != NULL){
            cpu_name = std::string(line);
        }
    }
    // cpu name concat board name, platform name and chip name
    char board_name[128];
    char platform_name[128];
    char chip_name[128];
    __system_property_get("ro.product.board", board_name);
    __system_property_get("ro.board.platform", platform_name);
    __system_property_get("ro.chipname", chip_name);
    cpu_name = cpu_name + "_" + board_name + "_" + platform_name + "_" + chip_name;
    std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
    LOG(INFO) << "CPU Name : " << cpu_name;
    fclose(fp);
    return cpu_name;
}


int get_max_freq_khz(int cpuid) {
    // first try, for all possible cpu
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",\
     cpuid);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        // second try, for online cpu
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",\
         cpuid);
        fp = fopen(path, "rb");
    }

    int max_freq_khz = 0;
    if (fp){
        while (!feof(fp)) {
            int freq_khz = 0;
            int nscan = fscanf(fp, "%d %*d", &freq_khz);
            if (nscan != 1) {
                break;
            }

            if (freq_khz > max_freq_khz) {
                max_freq_khz = freq_khz;
            }
        }
    }
    if (max_freq_khz == 0 || !fp){
        // third try, for online cpu
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",\
            cpuid);
        fp = fopen(path, "rb");
        if (!fp) {
            return -1;
        }
        int max_freq_khz = -1;
        fscanf(fp, "%d", &max_freq_khz);
        fclose(fp);
        return max_freq_khz;
    }

    fclose(fp);
    return max_freq_khz;
}

int arm_sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>& cpuids, \
           std::vector<int>& cpu_freq, std::vector<int>& cluster_ids) {

    if (cpu_count == 0) {
        return 0;
    }

    cpuids.resize(cpu_count);
    cluster_ids.resize(cpu_count);

    for (int i = 0; i < cpu_count; i++) {
        cpuids[i] = i;
    }

    // sort cpuid as big core first
    //simple bubble sort

    for (int i = 0; i < cpu_count; i++)
    {
        for (int j = i+1; j < cpu_count; j++)
        {
            if (cpu_freq[i] < cpu_freq[j])
            {
                // swap
                int tmp = cpuids[i];
                cpuids[i] = cpuids[j];
                cpuids[j] = tmp;
            }
        }
    }
    // SMP
    int mid_max_freq_khz = (cpu_freq[cpuids[0]] + cpu_freq[cpuids[cpu_count - 1]]) / 2;

    for (int i = 0; i < cpu_count; i++) {
        cpuids[i] = i;
        if (cpu_freq[i] >= mid_max_freq_khz) {
            cluster_ids[i] = 0;
        }
        else{
            cluster_ids[i] = 1;
        }
    }

    return 0;
}

int check_online(std::vector<int>& core_ids){

    if (core_ids.size() == 0){
        return 0;
    }
    char path[256];
    int online = 1;
    for (int i = 0; i < core_ids.size(); ++i){
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online",\
            core_ids[i]);
        FILE* fp = fopen(path, "rb");
        if (!fp){
            return 0;
        }
        int cur_online = 0;
        fscanf(fp, "%d", &cur_online);
        online &= cur_online;
        fclose(fp);
    }
    return online;
}

int set_sched_affinity(const std::vector<int>& cpuids) {
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
    pid_t pid = gettid();
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < cpuids.size(); i++) {
        CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        LOG(ERROR) << "syscall error" << syscallret;
        return -1;
    }

    return 0;
}
#endif //android

template <>
void Device<ARM>::create_stream() {
    _compute_stream.resize(_max_stream);
    _data_stream.resize(_max_stream);
    for (int i = 0; i < _max_stream; ++i) {
        _compute_stream[i] = nullptr;
        _data_stream[i] = nullptr;
    }
}

template <>
void Device<ARM>::get_info() {
    set_default_cache(_info);
    _info._compute_core_num = arm_get_cpucount();
    _info._max_memory = arm_get_meminfo();
    //get max freq
#ifdef PLATFORM_ANDROID
    std::vector<int> max_freq(_info._compute_core_num);
    for (int i = 0; i < _info._compute_core_num; ++i){
        max_freq[i] = get_max_freq_khz(i) / 1000;
    }
    std::string cpu_name = arm_get_cpu_name();
    if (get_cpu_info_from_name(_info, cpu_name) != SaberSuccess){
        arm_sort_cpuid_by_max_frequency(_info._compute_core_num, _info._core_ids, max_freq, _info._cluster_ids);
        _info._big_core_ids.clear();
        _info._little_core_ids.clear();
        for (int i = 0; i < _info._cluster_ids.size(); ++i) {
            if (_info._cluster_ids[i] == 0) {
                _info._big_core_ids.push_back(_info._core_ids[i]);
            } else {
                _info._little_core_ids.push_back(_info._core_ids[i]);
            }
        }
        arm_get_cpu_arch(_info._archs);
    }

    LOG(INFO) << "ARM multiprocessors number: " <<  _info._compute_core_num;
    for (int i = 0; i < _info._compute_core_num; ++i) {
        LOG(INFO) <<"ARM multiprocessors ID:" << _info._core_ids[i] << ", frequence:" << max_freq[i] << \
        ", cluster ID: " << _info._cluster_ids[_info._core_ids[i]] << ", CPU ARCH: " << _info._archs[i];
    }
    LOG(INFO) << "L1 Cache size is: ";
    if (_info._big_core_ids.size() > 0){
        LOG(INFO) << "big core: " << _info._L1_cache[_info._big_core_ids[0]] / 1024 << "KB";
    }
    if (_info._little_core_ids.size() > 0){
        LOG(INFO) << "little core: " << _info._L1_cache[_info._little_core_ids[0]] / 1024 << "KB";
    }
    LOG(INFO) << "L2 Cache size is: ";
    if (_info._big_core_ids.size() > 0){
        LOG(INFO) << "big core: " << _info._L2_cache[_info._big_core_ids[0]] / 1024 << "KB";
    }
    if (_info._little_core_ids.size() > 0){
        LOG(INFO) << "little core: " << _info._L2_cache[_info._little_core_ids[0]] / 1024 << "KB";
    }

    LOG(INFO) << "Total memory: " << _info._max_memory << "KB";
    _info._max_frequence = max_freq[0];
    for (int j = 1; j < _info._compute_core_num; ++j) {
        if (_info._max_frequence < max_freq[j]){
            _info._max_frequence = max_freq[j];
        }
    }
#elif defined(TARGET_IOS)
    arm_get_cpu_arch(_info._archs);
#endif
}

template <>
void Context<ARM>::bind_dev() {
#ifdef USE_OPENMP
    int num_threads = _act_ids.size();
    omp_set_num_threads(num_threads);
#ifdef PLATFORM_ANDROID
    std::vector<int> ssarets;
    for (int j = 0; j < num_threads; ++j) {
        ssarets.push_back(0);
    }
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ssarets[i] = set_sched_affinity(_act_ids);
    }
    for (int i = 0; i < num_threads; i++) {
        if (ssarets[i] != 0) {
            LOG(ERROR) << "set cpu affinity failed, cpuID: " <<  _act_ids[i];
            return;
        }
    }
#endif //PLATFORM_ANDROID
#else //USE_OPENMP
#ifdef PLATFORM_ANDROID
    std::vector<int> cpuid1;
    cpuid1.push_back(_act_ids[0]);
    int ssaret = set_sched_affinity(cpuid1);
    if (ssaret != 0) {
        printf("set cpu affinity failed, cpuID: %d\n", _act_ids[0]);
        return;
    }
#endif //PLATFORM_ANDROID
#endif//USE_OPENMP
}

template <>
void Context<ARM>::set_run_mode(PowerMode mode, int threads) {

    int big_core_size = devs[_device_id]._info._big_core_ids.size();
    int small_core_size = devs[_device_id]._info._little_core_ids.size();
#ifdef USE_OPENMP
    if (threads > big_core_size + small_core_size) {
        threads = big_core_size + small_core_size;
    }
    _count++;
    int shift_num = (_count / 10) % big_core_size;
    switch (mode) {
        case SABER_POWER_FULL:
            _mode = mode;
            _act_ids.clear();
            for (int i = 0; i < threads; ++i) {
                if (i < big_core_size) {
                    _act_ids.push_back(devs[_device_id]._info._big_core_ids[i]);
                } else {
                    _act_ids.push_back(devs[_device_id]._info._little_core_ids[i - big_core_size]);
                }
            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
        case SABER_POWER_HIGH:
            _act_ids.clear();
            if (big_core_size > 0) {
                _mode = SABER_POWER_HIGH;
                if (threads > big_core_size) {
                    LOG(ERROR) << "threads: " << threads << ", exceed the big cores size: " << big_core_size;
                    _act_ids = devs[_device_id]._info._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._big_core_ids[devs[_device_id]._info._big_core_ids.size() - 1 - i]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                LOG(ERROR) << "HIGH POWER MODE is not support, switch to little cores";
                if (threads > small_core_size) {
                    _act_ids = devs[_device_id]._info._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._little_core_ids[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
        case SABER_POWER_LOW:
            _act_ids.clear();
            if (small_core_size > 0) {
                _mode = SABER_POWER_LOW;
                if (threads > small_core_size) {
                    LOG(WARNING) << "threads: " << threads << ", exceed the little cores size:" << small_core_size;
                    _act_ids = devs[_device_id]._info._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._little_core_ids[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores";
                if (threads > big_core_size) {
                    _act_ids = devs[_device_id]._info._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._big_core_ids[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
        case SABER_POWER_NO_BIND:
            _mode = SABER_POWER_NO_BIND;
            _act_ids.clear();
            if (threads > devs[_device_id]._info._core_ids.size()) {
                _act_ids.resize(devs[_device_id]._info._core_ids.size());
            } else {
                _act_ids.resize(threads);
            }
            break;
        case SABER_POWER_RAND_HIGH:
            _act_ids.clear();
            if (big_core_size > 0) {
                _mode = SABER_POWER_RAND_HIGH;
                if (threads > big_core_size) {
                    LOG(WARNING) << "threads: " << threads << ", exceed the big cores size: " << big_core_size;
                    _act_ids = devs[_device_id]._info._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._big_core_ids[(i + shift_num) % big_core_size]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                LOG(WARNING) << "HIGH POWER MODE is not support, switch to little cores";
                if (threads > small_core_size) {
                    _act_ids = devs[_device_id]._info._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._little_core_ids[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
        case SABER_POWER_RAND_LOW:
            _act_ids.clear();
            if (small_core_size > 0) {
                _mode = SABER_POWER_RAND_LOW;
                if (threads > small_core_size) {
                    LOG(WARNING) << "threads: " << threads << ", exceed the little cores size: " << small_core_size;
                    _act_ids = devs[0]._info._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._little_core_ids[(i + shift_num) % small_core_size]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores";
                if (threads > big_core_size) {
                    _act_ids = devs[_device_id]._info._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(devs[_device_id]._info._big_core_ids[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
    }
    
    if (_mode == SABER_POWER_NO_BIND) {
        int threads = _act_ids.size();
        omp_set_num_threads(threads);
    } else {
        if (check_online(_act_ids)){
            bind_dev();
        } else {
            LOG(INFO) << "some cpu is offline, switch to NO BIND MODE";
            int threads = _act_ids.size();
            omp_set_num_threads(threads);
        }
    }
#else
    if (big_core_size > 0){
        _act_ids = {devs[_device_id]._info._big_core_ids[0]};
    } else {
        _act_ids = {0};
    }
#endif
    _arch = devs[_device_id]._info._archs[_act_ids[0]];
}

template <>
PowerMode Context<ARM>::get_mode() const{
    return _mode;
}
template <>
ARMArch Context<ARM>::get_arch() const{
    return _arch;
}
template <>
void Context<ARM>::set_arch(ARMArch arch) {
    _arch = arch;
}

template <>
void Context<ARM>::set_cache(int l1size, int l2size, int l3size) {
    int cpu_count = arm_get_cpucount();
    devs[_device_id]._info._L1_cache.resize(cpu_count);
    devs[_device_id]._info._L2_cache.resize(cpu_count);
    devs[_device_id]._info._L3_cache.resize(cpu_count);
    for (int i = 0;i < cpu_count; ++i){
        devs[_device_id]._info._L1_cache[i] = l1size;
        devs[_device_id]._info._L2_cache[i] = l2size;
        devs[_device_id]._info._L3_cache[i] = l3size;
    }
    int temp_mem_size = 2 * (l1size + l2size);
    _work_space.reshape(Shape({1, 1, 1, temp_mem_size}));
}

template<>
int Context<ARM>::get_l1_cache_size() const{
    return devs[_device_id]._info._L1_cache[_act_ids[0]];
}

template<>
int Context<ARM>::get_l2_cache_size() const{
    return devs[_device_id]._info._L2_cache[_act_ids[0]];
}

template<>
int Context<ARM>::get_l3_cache_size() const{
    return devs[_device_id]._info._L3_cache[_act_ids[0]];
}

template<>
void* Context<ARM>::get_work_space() {
    return (void*)_work_space.mutable_data();
}

template<>
int Context<ARM>::get_threads() const {
    return _act_ids.size();
}

template<>
SaberStatus Context<ARM>::workspace_extend(Shape sh) {
    int count = sh.count();
    Shape old = _work_space.shape();
    _work_space.reshape(Shape({1, 1, 1, count + devs[_device_id]._info._L2_cache[_act_ids[0]] / sizeof(float)}));

    if (_work_space.data() == nullptr) {
        _work_space.re_alloc(old, AK_FLOAT);
        return SaberInvalidValue;
    }
    return SaberSuccess;
}

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
