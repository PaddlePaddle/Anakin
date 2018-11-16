#include "saber/lite/core/context_lite.h"

#ifdef PLATFORM_ANDROID
#include <sys/syscall.h>
#include <unistd.h>
#endif
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

namespace lite{

int arm_get_cpucount(std::vector<ARMArch>& archs) {
#ifdef PLATFORM_ANDROID
    // get cpu count from /proc/cpuinfo
#if 0 //read from /proc/cpuinfo
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
#else
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
    archs.clear();
    //! get CPU ARCH
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
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
                case 0x800:
                    // 835
                    archs.push_back(A73);
                    break;
                case 0x205:
                    // 820
                    archs.push_back(A72);
                    break;
                default:
                    LOGE("ERROR: CPU ARCH unknow type\n");
                    archs.push_back(ARM_UNKOWN);
            }
        }
    }

    fclose(fp);
    if (archs.size() < count) {
        for (int i = archs.size(); i < count; ++i) {
            archs.push_back(archs[i - 1]);
        }
    }
    return count;
#endif //if 0

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
#endif //android
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
        LOGE("ERROR: syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}
#endif //android
//template <>
void Env::get_info(DeviceInfo& dev) {
    //! set to const value, need to fetch from device
#ifdef TARGET_IOS
    dev._L1_cache = 64 * 1024;
    dev._L2_cache = 2048 * 1024;
    dev._L3_cache = 0;
#else
    dev._L1_cache = 32 * 1024;
    dev._L2_cache = 512 * 1024;
    dev._L3_cache = 0;
#endif
    dev._compute_core_num = arm_get_cpucount(dev._archs);
    dev._max_memory = arm_get_meminfo();

    //_max_stream = _info._compute_core_num;
#ifdef PLATFORM_ANDROID
    std::vector<int> max_freq;

    arm_sort_cpuid_by_max_frequency(dev._compute_core_num, dev._core_ids, max_freq, dev._cluster_ids);

    dev._big_core_ids.clear();
    dev._little_core_ids.clear();
    for (int i = 0; i < dev._cluster_ids.size(); ++i) {
        if (dev._cluster_ids[i] == 0) {
            dev._big_core_ids.push_back(dev._core_ids[i]);
        } else {
            dev._little_core_ids.push_back(dev._core_ids[i]);
        }
    }
    LOGI("ARM multiprocessors number: %d\n", dev._compute_core_num);
    for (int i = 0; i < dev._compute_core_num; ++i) {
        LOGI("ARM multiprocessors ID: %d, frequence: %d, cluster ID: %d, CPU ARCH: A%d\n", \
                dev._core_ids[i], max_freq[dev._core_ids[i]], dev._cluster_ids[dev._core_ids[i]], \
                dev._archs[i]);
    }
    LOGI("L1 DataCache size Not Set, current is:  %d B\n", dev._L1_cache);
    LOGI("L2 Cache size Not Set, current is:  %d B\n", dev._L2_cache);
    LOGI("Total memory: %d kB\n", dev._max_memory);

    dev._max_frequence = max_freq[0];
    for (int j = 1; j < dev._compute_core_num; ++j) {
        if (dev._max_frequence < max_freq[j]){
            dev._max_frequence = max_freq[j];
        }
    }
#elif defined(TARGET_IOS)
    printf("ios target, unsupport now\n");
#endif
}

void Context::set_cache(size_t l1size, size_t l2size, size_t l3size) {
    DeviceInfo& dev = Env::cur_env();
    dev._L1_cache = l1size;
    dev._L2_cache = l2size;
    dev._L3_cache = l3size;
    int temp_mem_size = 2 * (l1size + l2size);
    _work_space.reshape(Shape(temp_mem_size));
}

//template <>
Context::Context() {
    //! 1 thread, big core
    _act_ids = {0};
    _mode = SABER_POWER_HIGH;
    DeviceInfo& dev = Env::cur_env();
#ifdef TARGET_IOS
    _arch = APPLE; //use 6x8
#else
    _arch = dev._archs[dev._big_core_ids[0]];
#endif
}

PowerMode Context::get_mode() const {
    return _mode;
}

int Context::get_threads() const {
    return _act_ids.size();
}

Context::Context(const Context& ctx){
    _mode = ctx._mode;
    _act_ids = ctx._act_ids;
    _work_space = ctx._work_space;
    _arch = ctx._arch;
    _count = ctx._count;
}

Context& Context::operator=(const Context &ctx) {
    _mode = ctx._mode;
    _act_ids = ctx._act_ids;
    _work_space = ctx._work_space;
    _arch = ctx._arch;
    _count = ctx._count;
    return *this;
}

void Context::bind_dev() {
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
            LOGE("ERROR: set cpu affinity failed, cpuID: %d\n", _act_ids[i]);
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

void Context::set_run_mode(PowerMode mode, int threads) {
#ifdef USE_OPENMP
    DeviceInfo& dev = Env::cur_env();
    int big_core_size = dev._big_core_ids.size();
    int small_core_size = dev._little_core_ids.size();
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
                    _act_ids.push_back(dev._big_core_ids[i]);
                } else {
                    _act_ids.push_back(dev._little_core_ids[i - big_core_size]);
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
                    LOGE("ERROR: threads: %d, exceed the big cores size: %d\n", threads, big_core_size);
                    _act_ids = dev._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._big_core_ids[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                LOGE("ERROR: HIGH POWER MODE is not support, switch to small cores\n");
                if (threads > small_core_size) {
                    _act_ids = dev._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[i]);
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
                    LOGW("threads: %d, exceed the small cores size: %d\n", threads, small_core_size);
                    _act_ids = dev._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                LOGW("LOW POWER MODE is not support, switch to big cores\n");
                if (threads > big_core_size) {
                    _act_ids = dev._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[i]);
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
            if (threads > dev._core_ids.size()) {
                _act_ids.resize(dev._core_ids.size());
            } else {
                _act_ids.resize(threads);
            }
            break;
        case SABER_POWER_RAND_HIGH:
            _act_ids.clear();
            if (big_core_size > 0) {
                _mode = SABER_POWER_RAND_HIGH;
                if (threads > big_core_size) {
                    LOGW("threads: %d, exceed the big cores size: %d\n", threads, big_core_size);
                    _act_ids = dev._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._big_core_ids[(i + shift_num) % big_core_size]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                LOGW("HIGH POWER MODE is not support, switch to small cores\n");
                if (threads > small_core_size) {
                    _act_ids = dev._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[i]);
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
                    LOGW("threads: %d, exceed the small cores size: %d\n", threads, small_core_size);
                    _act_ids = dev._little_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[(i + shift_num) % small_core_size]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                LOGW("LOW POWER MODE is not support, switch to big cores\n");
                if (threads > big_core_size) {
                    _act_ids = dev._big_core_ids;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(dev._little_core_ids[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
    }
#ifdef TARGET_IOS
    _arch = APPLE; //use 6x8
#else
//    if (_mode == SABER_POWER_RAND_LOW || _mode == SABER_POWER_LOW) {
//        _arch = dev._archs[dev._little_core_ids[0]];
//    } else {
//        _arch = dev._archs[dev._big_core_ids[0]];
//    }
#endif
//    LOGI("run mode: %d, arch: %d\n", _mode, _arch);
//    LOGI("thread num: %lu\n", _act_ids.size());
//    for (int j = 0; j < _act_ids.size(); ++j) {
//        LOGI("|----active id: %d\n", _act_ids[j]);
//    }
    // fixme
    //! fix multi-threads SABER_POWER_HIGH mode
    if (_mode == SABER_POWER_NO_BIND || _mode == SABER_POWER_HIGH) {
        int threads = _act_ids.size();
        omp_set_num_threads(threads);
    } else {
        bind_dev();
    }
#else
    _act_ids = {0};
#endif
    //! alloc memory for sgemm in this context
    int temp_mem_size = Env::cur_env()._L2_cache / sizeof(float);
    _work_space.reshape(Shape(temp_mem_size));
}

void* Context::get_work_space() {
    return (void*)_work_space.mutable_data();
}

ARMArch Context::get_arch() const {
    return _arch;
}

void Context::set_arch(ARMArch arch) {
    _arch = arch;
}

size_t Context::l1_cache_size() const {
    DeviceInfo& dev = Env::cur_env();
    return dev._L1_cache;
}

size_t Context::l2_cache_size() const {
    DeviceInfo& dev = Env::cur_env();
    return dev._L2_cache;
}

void Context::workspace_extend(Shape sh) {
    int count = sh.count();
    _work_space.reshape(Shape(count + l2_cache_size() / sizeof(float)));
}

} //namespace lite

} //namespace saber

} //namespace anakin

