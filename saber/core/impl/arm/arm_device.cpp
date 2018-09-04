#include "device.h"
#include "context.h"

#ifdef USE_ARM_PLACE

#ifdef PLATFORM_ANDROID
#include <sys/syscall.h>
#include <unistd.h>
#define __NCPUBITS__  (8 * sizeof (unsigned long))

#define __CPU_SET(cpu, cpusetp) \
  ((cpusetp)->mask_bits[(cpu) / __NCPUBITS__] |= (1UL << ((cpu) % __NCPUBITS__)))

#define __CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))

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

#elif TARGET_IOS
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

size_t arm_get_meminfo() {
#ifdef PLATFORM_ANDROID
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/meminfo", "rb");
    if (!fp) {
        return 1;
    }

    size_t memsize = 0;
    char line[1024];
    while (!feof(fp))
    {
        char* s = fgets(line, 1024, fp);
        if (!s) {
            break;
        }
        sscanf(s, "MemTotal:        %d kB", &memsize);
    }

    fclose(fp);

    return memsize;
#elif TARGET_IOS
    // to be implemented
    LOG(ERROR) << "not implemented";
    return 0;
#endif
}

#ifdef PLATFORM_ANDROID
static int get_max_freq_khz(int cpuid)
{
    // first try, for all possible cpu
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",\
     cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
    {
        // second try, for online cpu
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",\
         cpuid);
        fp = fopen(path, "rb");

        if (!fp)
        {
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
    while (!feof(fp))
    {
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

int arm_sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>& cpuids, \
           std::vector<int>& cpu_freq, std::vector<int>& cluster_ids) {
    //const int cpu_count = cpuids.size();

    if (cpu_count == 0) {
        return 0;
    }

    //std::vector<int> cpu_max_freq_khz;
    cpuids.resize(cpu_count);
    cpu_freq.resize(cpu_count);
    cluster_ids.resize(cpu_count);

    for (int i = 0; i < cpu_count; i++)
    {
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

int set_sched_affinity(const std::vector<int>& cpuids) {
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity

    typedef struct {
        unsigned long mask_bits[1024 / __NCPUBITS__];
    }cpu_set_t;

    // set affinity for thread
    pid_t pid = gettid();

    cpu_set_t mask;
    __CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuids.size(); i++)
    {
        __CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret)
    {
        LOG(ERROR) << "syscall error " << syscallret;
        return -1;
    }

    return 0;
}

int set_cpu_affinity(const std::vector<int>& cpuids) {
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
            LOG(ERROR)<<"set cpu affinity failed, cpuID: " << cpuids[i];
            return -1;
        }
    }
#else
    std::vector<int> cpuid1;
    cpuid1.push_back(cpuids[0]);
    int ssaret = set_sched_affinity(cpuid1);
        if (ssaret != 0) {
            LOG(ERROR)<<"set cpu affinity failed, cpuID: " << cpuids[0];
            return -1;
        }
#endif
    return 0;
}
#endif //PLATFORM_ANDROID

#ifdef TARGET_IOS
int set_cpu_affinity(const std::vector<int>& cpuids) {
#ifdef USE_OPENMP
    int num_threads = cpuids.size();
    omp_set_num_threads(num_threads);
#endif
    return 0;
}
#endif

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

    //! set to const value, need to fetch from device
    _info._L1_cache = 31000;
    _info._L2_cache = 2000000;
    _info._L3_cache = 0;

    _info._idx = 0;
    _info._compute_core_num = arm_get_cpucount();
    _info._max_memory = arm_get_meminfo();

    _max_stream = _info._compute_core_num;

    std::vector<int> max_freq;

    arm_sort_cpuid_by_max_frequency(_info._compute_core_num, _info._core_ids, max_freq, _info._cluster_ids);

    LOG(INFO) << "ARM multiprocessors number: " << _info._compute_core_num;
    for (int i = 0; i < _info._compute_core_num; ++i) {
        LOG(INFO) << "ARM multiprocessors ID: " << _info._core_ids[i] \
            << ", frequence: " << max_freq[_info._core_ids[i]] << " MHz" << \
            ", cluster ID: " << _info._cluster_ids[_info._core_ids[i]];
    }
    //LOG(INFO) << "L1 DataCache size: " << L1_cache << "B";
    //LOG(INFO) << "L2 Cache size: " << L2_cache << "B";
    LOG(INFO) << "Total memory: " << _info._max_memory << "kB";

    _info._max_frequence = max_freq[0];
    for (int j = 1; j < _info._compute_core_num; ++j) {
        if(_info._max_frequence < max_freq[j]){
            _info._max_frequence = max_freq[j];
        }
    }
}

template <>
void Context<ARM>::bind_dev() {
    set_cpu_affinity(_act_ids);
}

template <>
void Context<ARM>::set_run_mode(PowerMode mode, int threads) {
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < devs[0]._info._cluster_ids.size(); ++i) {
        if (devs[0]._info._cluster_ids[i] == 0) {
            big_cores.push_back(devs[0]._info._core_ids[i]);
        } else {
            small_cores.push_back(devs[0]._info._core_ids[i]);
        }
    }
    int big_core_size = big_cores.size();
    int small_core_size = small_cores.size();
    if (threads > big_core_size + small_core_size) {
        threads = big_core_size + small_core_size;
    }
    switch (mode) {
        case SABER_POWER_FULL:
            _mode = mode;
            _act_ids.clear();
            for (int i = 0; i < threads; ++i) {
                if (i < big_core_size) {
                    _act_ids.push_back(big_cores[i]);
                } else {
                    _act_ids.push_back(small_cores[i - big_core_size]);
                }
            }
            break;
        case SABER_POWER_HIGH:
            _act_ids.clear();
            if (big_core_size > 0) {
                _mode = SABER_POWER_HIGH;
                if (threads > big_core_size) {
                    LOG(ERROR) << "threads: " << threads << " exceed the big cores size: " << big_core_size;
                    _act_ids = big_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(big_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                LOG(ERROR) << "HIGH POWER MODE is not support, switch to small cores";
                if(threads > small_core_size) {
                    _act_ids = small_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }

            }
            break;
        case SABER_POWER_LOW:
            _act_ids.clear();
            if (small_core_size > 0) {
                _mode = SABER_POWER_LOW;
                if (threads > small_core_size) {
                    LOG(ERROR) << "threads: " << threads << " exceed the small cores size: " << small_core_size;
                    _act_ids = small_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                LOG(ERROR) << "LOW POWER MODE is not support, switch to big cores";
                if(threads > big_core_size) {
                    _act_ids = big_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }

            }
            break;
    }
    LOG(INFO) << "mode: \n0: big cores only;\n1: small cores only;\n2: all cores";
    LOG(INFO) << "|----run mode: " << 0;
    LOG(INFO) << "|----thread num: " << _act_ids.size();
    for (int j = 0; j < _act_ids.size(); ++j) {
        LOG(INFO) << "|----active id: " << _act_ids[j];
    }
    bind_dev();
}

template <>
PowerMode Context<ARM>::get_mode(int& threads) {
    threads = _act_ids.size();
    return _mode;
}

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE