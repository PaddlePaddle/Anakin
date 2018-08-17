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

#ifndef ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
#define ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
#include "saber/lite/core/common_lite.h"
#include "saber/lite/core/tensor_lite.h"
namespace anakin{

namespace saber{

namespace lite{

struct DeviceInfo{
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

//template <ARMType ttype>
//class Device {
//public:
//    Device(int max_stream = 4);
//    void get_info();
//    void create_stream();
//
//    DeviceInfo _info;
//    int _max_stream;
//    std::vector<typename TargetTrait<ttype>::stream_t> _data_stream;
//    std::vector<typename TargetTrait<ttype>::stream_t> _compute_stream;
//};


//template <ARMType ttype>
class Env {
public:
    //typedef std::vector<Device<ttype>> Devs;
    static DeviceInfo& cur_env() {
        static DeviceInfo* _g_env = new DeviceInfo();
        return *_g_env;
    }
    static void env_init(int max_stream = 4) {
        DeviceInfo& devs = cur_env();
        get_info(devs);
    }

private:
    static void get_info(DeviceInfo& dev);
    Env(){}
};

//template <ARMType ttype>
class Context {
public:
    Context();
    /**
     * \brief context constructor, set device id, data stream id and compute stream id
     * @param device_id
     * @param data_stream_id
     * @param compute_stream_id
     */
    Context(PowerMode mode, int threads);

    Context(const Context& ctx);

    Context&operator=(const Context& ctx);

    void set_run_mode(PowerMode mode, int threads);
    void bind_dev();
    PowerMode get_mode();
    int get_threads();
    void set_cache(size_t l1size, size_t l2size, size_t l3size);
    void* get_work_space();
private:

    //! SABER_POWER_HIGH stands for using big cores,
    //! SABER_POWER_LOW stands for using small core,
    //! SABER_POWER_FULL stands for using all cores
    PowerMode _mode;
    std::vector<int> _act_ids;
    Tensor<CPU, AK_FLOAT> _work_space;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_DEVICE_LITE_H
