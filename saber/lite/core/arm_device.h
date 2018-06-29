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

#ifndef ANAKIN_SABER_LITE_CORE_ARM_DEVICE_H
#define ANAKIN_SABER_LITE_CORE_ARM_DEVICE_H

#include <stdio.h>
#include <vector>

#ifdef PLATFORM_ANDROID
#include <sys/syscall.h>
#include <unistd.h>

#define __NCPUBITS__  (8 * sizeof (unsigned long))

#define __CPU_SET(cpu, cpusetp) \
  ((cpusetp)->mask_bits[(cpu) / __NCPUBITS__] |= (1UL << ((cpu) % __NCPUBITS__)))

#define __CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))
#endif

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#define __IOS__
#endif
#endif

#ifdef USE_ARM_PLACE
static int arm_get_cpucount() {
#ifdef PLATFORM_ANDROID
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }

    int count = 0;
    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s) {
            break;
        }

        if (memcmp(line, "processor", 9) == 0) {
            count++;
        }
    }

    fclose(fp);

    if (count < 1) {
        count = 1;
    }

    return count;
#elif __IOS__
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

static int arm_get_meminfo() {
#ifdef PLATFORM_ANDROID
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/meminfo", "rb");
    if (!fp) {
        return 1;
    }

    int memsize = 0;
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
#elif __IOS__
    // to be implemented
    return 0;
#endif
}

#ifdef PLATFORM_ANDROID
static int get_max_freq_khz(int cpuid) {
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

        if (!fp) {
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
    }

    int max_freq_khz = 0;
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

    fclose(fp);

    return max_freq_khz;
}

static int arm_sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>& cpuids, \
           std::vector<int>& cpu_freq, std::vector<int>& cluster_ids) {
    //const int cpu_count = cpuids.size();

    if (cpu_count == 0) {
        return 0;
    }

    //std::vector<int> cpu_max_freq_khz;
    cpuids.resize(cpu_count);
    cpu_freq.resize(cpu_count);
    cluster_ids.resize(cpu_count);

    for (int i = 0; i < cpu_count; i++) {
        int max_freq_khz = get_max_freq_khz(i);
        //printf("%d max freq = %d khz\n", i, max_freq_khz);
        cpuids[i] = i;
        cpu_freq[i] = max_freq_khz / 1000;
    }

    // SMP
    int mid_max_freq_khz = (cpu_freq.front() + cpu_freq.back()) / 2;

    for (int i = 0; i < cpu_count; i++) {
        if (cpu_freq[i] >= mid_max_freq_khz) {
            cluster_ids[i] = 0;
        }
        else{
            cluster_ids[i] = 1;
        }
    }

    return 0;
}

#endif // __ANDROID__

#ifdef __IOS__
static int sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>& cpuids, \
       std::vector<int>& cpu_freq, std::vector<int>& cluster_ids){
    if (cpu_count == 0) {
        return 0;
    }
    cpuids.resize(cpu_count);
    cpu_freq.resize(cpu_count);
    cluster_ids.resize(cpu_count);
    for (int i = 0; i < cpu_count; ++i) {
        cpuids[i] = i;
        cpu_freq[i] = 1000;
        cluster_ids[i] = 0;
    }
}
#endif

#ifdef PLATFORM_ANDROID
static int set_sched_affinity(const std::vector<int>& cpuids) {
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity

    typedef struct {
        unsigned long mask_bits[1024 / __NCPUBITS__];
    } cpu_set_t;

    // set affinity for thread
    pid_t pid = gettid();

    cpu_set_t mask;
    __CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuids.size(); i++) {
        __CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        //LOG(ERROR) << "syscall error " << syscallret;
        return -1;
    }

    return 0;
}

static int set_cpu_affinity(const std::vector<int>& cpuids){
#ifdef USE_OPENMP
    int num_threads = cpuids.size();
    omp_set_num_threads(num_threads);
    std::vector<int> ssarets(num_threads, 0);
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ssarets[i] = set_sched_affinity(cpuids);
    }
    for (int i = 0; i < num_threads; i++) {
        if (ssarets[i] != 0) {
            //LOG(ERROR)<<"set cpu affinity failed, cpuID: " << cpuids[i];
            return -1;
        }
    }
#else
    std::vector<int> cpuid1;
    cpuid1.push_back(cpuids[0]);
    int ssaret = set_sched_affinity(cpuid1);
        if (ssaret != 0) {
            //LOG(ERROR)<<"set cpu affinity failed, cpuID: " << cpuids[0];
            return -1;
        }
#endif
}

#endif //PLATFORN_ANDROID

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_CORE_ARM_DEVICE_H
