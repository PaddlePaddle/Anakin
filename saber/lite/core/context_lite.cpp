#include "saber/lite/core/context_lite.h"

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

namespace lite{

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
#elif defined(TARGET_IOS)
    // to be implemented
    printf("not implemented\n");
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
    for (int i = 0; i < (int)cpuids.size(); i++) {
        __CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        printf("syscall error %d\n", syscallret);
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
            printf("set cpu affinity failed, cpuID: %d\n", cpuids[i]);
            return -1;
        }
    }
#else
    std::vector<int> cpuid1;
    cpuid1.push_back(cpuids[0]);
    int ssaret = set_sched_affinity(cpuid1);
        if (ssaret != 0) {
            printf("set cpu affinity failed, cpuID: %d\n", cpuids[0]);
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
#endif //TARGET_IOS


//template <>
void Env::get_info(DeviceInfo& dev) {
    //! set to const value, need to fetch from device
    dev._L1_cache = 31000;
    dev._L2_cache = 2000000;
    dev._L3_cache = 0;

    dev._compute_core_num = arm_get_cpucount();
    dev._max_memory = arm_get_meminfo();

    //_max_stream = _info._compute_core_num;
#ifdef PLATFORM_ANDROID
    std::vector<int> max_freq;

    arm_sort_cpuid_by_max_frequency(dev._compute_core_num, dev._core_ids, max_freq, dev._cluster_ids);

    printf("ARM multiprocessors number: %d\n", dev._compute_core_num);
    for (int i = 0; i < dev._compute_core_num; ++i) {
        printf("ARM multiprocessors ID: %d, frequence: %d, cluster ID: %d\n", \
                dev._core_ids[i], max_freq[dev._core_ids[i]], dev._cluster_ids[dev._core_ids[i]]);
    }
    //LOG(INFO) << "L1 DataCache size: " << L1_cache << "B";
    //LOG(INFO) << "L2 Cache size: " << L2_cache << "B";
    printf("Total memory: %d kB\n", dev._max_memory);

    dev._max_frequence = max_freq[0];
    for (int j = 1; j < dev._compute_core_num; ++j) {
        if(dev._max_frequence < max_freq[j]){
            dev._max_frequence = max_freq[j];
        }
    }
#elif defined(TARGET_IOS)
    printf("ios target, unsupport now\n");
#endif
}
#if 0
template <>
void Device<CPU>::create_stream() {
    _compute_stream.resize(_max_stream);
    _data_stream.resize(_max_stream);
    for (int i = 0; i < _max_stream; ++i) {
        _compute_stream[i] = nullptr;
        _data_stream[i] = nullptr;
    }
}

template <>
Device<CPU>::Device(int max_stream){
    _max_stream = max_stream;
    get_info();
    create_stream();
}
#endif

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
}

PowerMode Context::get_mode(int& threads) {
    threads = _act_ids.size();
    return _mode;
}

Context::Context(const Context& ctx){
    _mode = ctx._mode;
    _act_ids = ctx._act_ids;
    _work_space = ctx._work_space;
}

Context& Context::operator=(const Context &ctx) {
    _mode = ctx._mode;
    _act_ids = ctx._act_ids;
    _work_space = ctx._work_space;
    return *this;
}

void Context::bind_dev() {
    set_cpu_affinity(_act_ids);
}

void Context::set_run_mode(PowerMode mode, int threads) {
    DeviceInfo& dev = Env::cur_env();
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < dev._cluster_ids.size(); ++i) {
        if (dev._cluster_ids[i] == 0) {
            big_cores.push_back(dev._core_ids[i]);
        } else {
            small_cores.push_back(dev._core_ids[i]);
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
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
        case SABER_POWER_HIGH:
            _act_ids.clear();
            if (big_core_size > 0) {
                _mode = SABER_POWER_HIGH;
                if (threads > big_core_size) {
                    printf("threads: %d, exceed the big cores size: %d\n", threads, big_core_size);
                    _act_ids = big_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(big_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                printf("HIGH POWER MODE is not support, switch to small cores\n");
                if(threads > small_core_size) {
                    _act_ids = small_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
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
                    printf("threads: %d, exceed the small cores size: %d\n", threads, small_core_size);
                    _act_ids = small_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                printf("LOW POWER MODE is not support, switch to big cores\n");
                if(threads > big_core_size) {
                    _act_ids = big_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }

            }
            if (_act_ids.size() == 0) {
                _act_ids.push_back(0);
            }
            break;
    }
    printf("run mode: %d\n", _mode);
    printf("thread num: %lu\n", _act_ids.size());
    for (int j = 0; j < _act_ids.size(); ++j) {
        printf("|----active id: %d\n", _act_ids[j]);
    }

    //! alloc memory for sgemm in this context

    int temp_mem_size = 2 * (Env::cur_env()._L1_cache + Env::cur_env()._L2_cache);
    _work_space.reshape(Shape(temp_mem_size));
    bind_dev();
}

void* Context::get_work_space() {
    return (void*)_work_space.mutable_data();
}

#if 0
template <>
Context<CPU>::Context(int device_id, int data_stream_id, int compute_stream_id) {
    typename Env<CPU>::Devs& devs = Env<CPU>::cur_env();
    LCHECK_GT(devs.size(), 0, "Env is not initialized or current target is not exit!");
    if (device_id >= devs.size()){
        printf("device index exceeds the number of devices, set to default device(0)!\n");
        _device_id = 0;
    } else {
        _device_id = device_id;
    }
    if (data_stream_id >= devs[_device_id]._max_stream) {
        printf("data stream index exceeds the maximum stream number, set to default stream(0)!\n");
        data_stream_id = 0;
    }
    _stream_data = devs[_device_id]._data_stream[data_stream_id];
    _data_stream_id = data_stream_id;

    if (compute_stream_id >= devs[_device_id]._max_stream) {
        printf("compute stream index exceeds the maximum stream number, set to default stream(0)!\n");
        compute_stream_id = 0;
    }
    _stream_compute = devs[_device_id]._compute_stream[compute_stream_id];
    _compute_stream_id = compute_stream_id;
    _act_ids = {0};
    _mode = SABER_POWER_HIGH;
}

template <>
Context<CPU>::Context(const Context<CPU>& ctx){
    _device_id = ctx._device_id;
    _data_stream_id = ctx._data_stream_id;
    _compute_stream_id = ctx._compute_stream_id;
    _stream_compute = ctx._stream_compute;
    _stream_data = ctx._stream_data;
    _mode = ctx._mode;
    _act_ids = ctx._act_ids;
}

template <>
Context<CPU>& Context<CPU>::operator=(const Context<CPU>& ctx){
    this->_device_id = ctx._device_id;
    this->_data_stream_id = ctx._data_stream_id;
    this->_compute_stream_id = ctx._compute_stream_id;
    this->_stream_data = ctx._stream_data;
    this->_stream_compute = ctx._stream_compute;
    return *this;
}

template<>
bool Context<CPU>::operator==(const Context<CPU> &right) {
    bool comp_eq = true;
    comp_eq = comp_eq && (_device_id == right._device_id);
    comp_eq = comp_eq && (_data_stream_id == right._data_stream_id);
    comp_eq = comp_eq && (_compute_stream_id == right._compute_stream_id);
    return comp_eq;
}

/**
 * \brief get device id of current context
 * @return
 */
template <>
int Context<CPU>::get_device_id() {
    return _device_id;
}

/**
 * \brief get data process stream
 * @return
 */
template <>
typename TargetTrait<CPU>::stream_t Context<CPU>::get_data_stream(){
    return _stream_data;
}

/**
 * \brief get compute process stream
 * @return
 */
template <>
typename TargetTrait<CPU>::stream_t Context<CPU>::get_compute_stream(){
    return _stream_compute;
}

template <>
void Context<CPU>::bind_dev() {
    set_cpu_affinity(_act_ids);
}

template <>
void Context<CPU>::set_run_mode(PowerMode mode, int threads) {
    typename Env<CPU>::Devs& devs = Env<CPU>::cur_env();
    Device<CPU> dev = devs[_device_id];
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < dev._info._cluster_ids.size(); ++i) {
        if (dev._info._cluster_ids[i] == 0) {
            big_cores.push_back(dev._info._core_ids[i]);
        } else {
            small_cores.push_back(dev._info._core_ids[i]);
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
                    printf("threads: %d, exceed the big cores size: %d\n", threads, big_core_size);
                    _act_ids = big_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(big_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_LOW;
                printf("HIGH POWER MODE is not support, switch to small cores\n");
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
                    printf("threads: %d, exceed the small cores size: %d\n", threads, small_core_size);
                    _act_ids = small_cores;
                } else {
                    for (int i = 0; i < threads; ++i) {
                        _act_ids.push_back(small_cores[i]);
                    }
                }
            } else {
                _mode = SABER_POWER_HIGH;
                printf("LOW POWER MODE is not support, switch to big cores\n");
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

    bind_dev();
}
//
//void set_act_cores(std::vector<int> ids) {
//    Device dev = devs[_device_id];
//    if (ids.size() == 0){
//        _act_ids.resize(1);
//        _act_ids[0] = dev._info._core_ids[0];
//    }else {
//        _act_ids.clear();
//        for (int i = 0; i < ids.size(); ++i) {
//            if (ids[i] < dev._info._core_ids.size()){
//                _act_ids.push_back(ids[i]);
//            }
//        }
//    }
//    bind_dev();
//}

template <>
PowerMode Context<CPU>::get_mode(int& threads) {
    threads = _act_ids.size();
    return _mode;
}
template <>
std::vector<int> Context<CPU>::get_act_ids() {
    return _act_ids;
}
#endif

} //namespace lite

} //namespace saber

} //namespace anakin

