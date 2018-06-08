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

#ifndef ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
#define ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
#include "saber/lite/core/common_lite.h"
#include "saber/lite/core/arm_device.h"
namespace anakin{

namespace saber{

namespace lite{

//template <ARM_TYPE target>
struct ARM_API{
    typedef void* stream_t;
    typedef void* event_t;
};

//template <typename TargetType>
struct DeviceInfo{
	int _idx;
	std::string _device_name;
	int _max_frequence;
	int _min_frequence;
	int _generate_arch;
	int _compute_core_num;
	int _max_memory;
    int _sharemem_size;
    int _L1_cache;
    int _L2_cache;
    int _L3_cache;
	std::vector<int> _core_ids;
	std::vector<int> _cluster_ids;
};

//template <typename TargetType>
struct Device {

    Device(int max_stream = 4) : _max_stream(max_stream){
    	get_info();
    	create_stream();
    }
	static void get_device_count(int& count) {
        count = 1;
    }
	static int get_device_id() {
        return 0;
    }
	static void set_device(int id) {}
    void get_info() {
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
    void create_stream() {
        _compute_stream.resize(_max_stream);
        _data_stream.resize(_max_stream);
        for (int i = 0; i < _max_stream; ++i) {
            _compute_stream[i] = nullptr;
            _data_stream[i] = nullptr;
        }
    }
    DeviceInfo _info;
	int _max_stream;

    std::vector<typename ARM_API::stream_t> _data_stream;
    std::vector<typename ARM_API::stream_t> _compute_stream;
};


//template <typename TargetType>
class Env {
public:
    typedef std::vector<Device> Devs;
    static Devs& cur_env() {
        static Devs* _g_env = new Devs();
        return *_g_env;
    }
    static void env_init(int max_stream = 4){
        Devs& devs = cur_env();
        if (devs.size() > 0){
            return;
        }
        int count = 0;
        Device::get_device_count(count);
        if (count == 0) {
            LOG(WARNING) << "no device found!";
        } else {
            LOG(INFO) << "found " << count << " device(s)";
        }
        int cur_id = Device::get_device_id();
        for (int i = 0; i < count; i++) {
            Device::set_device(i);
            devs.push_back(Device(max_stream));
        }
        Device::set_device(cur_id);
    }
private:
    Env(){}
};

//template <typename TargetType>
class Context final{
public:
    typename Env::Devs& devs = Env::cur_env();
    /**
     * \brief context constructor, set device id, data stream id and compute stream id
     * @param device_id
     * @param data_stream_id
     * @param compute_stream_id
     */
    Context(int device_id = 0, int data_stream_id = 0, int compute_stream_id = 0){
        CHECK_GT(devs.size(), 0) << "Env is not initialized or current target is not exit!";
        if (device_id >= devs.size()){
            LOG(WARNING) << "device index exceeds the number of devices, set to default device(0)!";
            _device_id = 0;
        } else {
            _device_id = device_id;
        }
        if (data_stream_id >= devs[_device_id]._max_stream) {
            LOG(WARNING) << "data stream index exceeds the maximum stream number, set to default stream(0)!";
            data_stream_id = 0;
        }
        _stream_data = devs[_device_id]._data_stream[data_stream_id];
        _data_stream_id = data_stream_id;

        if (compute_stream_id >= devs[_device_id]._max_stream) {
            LOG(WARNING) << "compute stream index exceeds the maximum stream number, set to default stream(0)!";
            compute_stream_id = 0;
        }
        _stream_compute = devs[_device_id]._compute_stream[compute_stream_id];
        _compute_stream_id = compute_stream_id;
    }

    Context(const Context& ctx){
        _device_id = ctx._device_id;
        _data_stream_id = ctx._data_stream_id;
        _compute_stream_id = ctx._compute_stream_id;
        _stream_compute = ctx._stream_compute;
        _stream_data = ctx._stream_data;
    }

    Context& operator=(const Context& ctx){
        this->_device_id = ctx._device_id;
        this->_data_stream_id = ctx._data_stream_id;
        this->_compute_stream_id = ctx._compute_stream_id;
        this->_stream_data = ctx._stream_data;
        this->_stream_compute = ctx._stream_compute;
        return *this;
    }

    bool operator==(const Context &right) {
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
    int get_device_id() {
        return _device_id;
    }

    /**
     * \brief get data process stream
     * @return
     */
    typename ARM_API::stream_t get_data_stream(){
        return _stream_data;
    }

    /**
     * \brief get compute process stream
     * @return
     */
    typename ARM_API::stream_t get_compute_stream(){
        return _stream_compute;
    }

    void set_power_mode(PowerMode mode) {
        _mode = mode;
        Device dev = devs[_device_id];
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
    void set_act_cores(std::vector<int> ids) {
        Device dev = devs[_device_id];
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
    }
    void bind_dev() {
        set_cpu_affinity(_act_ids);
    }
    PowerMode get_mode() {
        return _mode;
    }
    std::vector<int> get_act_ids() {
        return _act_ids;
    }


private:
    //! current stream to process
    typename ARM_API::stream_t _stream_data;
    typename ARM_API::stream_t _stream_compute;
    //! current device id
    int _device_id;
    int _data_stream_id;
    int _compute_stream_id;
    PowerMode _mode;
    std::vector<int> _act_ids;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
