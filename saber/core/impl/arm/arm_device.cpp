#include "device.h"
#include "context.h"
#ifdef USE_ARM_PLACE
#include "arm_device.h"

namespace anakin{

namespace saber{

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

template <>
PowerMode Context<ARM>::get_mode(int& threads) {
    threads = _act_ids.size();
    return _mode;
}

#if 0
template <>
void Context<ARM>::set_power_mode(PowerMode mode) {
    _mode = mode;
    Device<ARM> dev = devs[_device_id];
    if (mode == SABER_POWER_FULL){
        _act_ids = dev._info._core_ids;
    }
    else if (mode == SABER_POWER_LOW) {
        _act_ids.clear();
        for (int i = 0; i < dev._info._cluster_ids.size(); ++i) {
            if (dev._info._cluster_ids[i] == 1) {
                _act_ids.push_back(dev._info._core_ids[i]);
            }
        }
        if (_act_ids.size() == 0){
            LOG(WARNING) << "LOW POWER MODE is not support";
            _act_ids.push_back(dev._info._core_ids[0]);
        }
    }
    else if (mode == SABER_POWER_HIGH){
        _act_ids.clear();
        for (int i = 0; i < dev._info._cluster_ids.size(); ++i) {
            if (dev._info._cluster_ids[i] == 0) {
                _act_ids.push_back(dev._info._core_ids[i]);
            }
        }
        if (_act_ids.size() == 0){
            LOG(WARNING) << "HIGH POWER MODE is not support";
            _act_ids.push_back(dev._info._core_ids[0]);
        }
    }
    bind_dev();
}

template <>
void Context<ARM>::set_act_cores(std::vector<int> ids) {

#ifdef USE_OPENMP
    int dynamic_current = 0;
    int num_threads_current = 1;
    dynamic_current = omp_get_dynamic();
    num_threads_current = omp_get_num_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(ids.size());
    _act_ids = ids;
#endif

#if 0
    Device<ARM> dev = devs[_device_id];
    if (ids.size() == 0){
        _act_ids.resize(1);
        _act_ids[0] = dev._info._core_ids[0];
    }else {
        _act_ids.clear();
        for (int i = 0; i < ids.size(); ++i) {
            if (ids[i] < dev._info._core_ids.size()){
                _act_ids.push_back(ids[i]);
            }
        }
    }
    bind_dev();
#endif
}

template <>
PowerMode Context<ARM>::get_mode() {
    return _mode;
}

template <>
std::vector<int> Context<ARM>::get_act_ids() {
    return _act_ids;
}
#endif


} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE