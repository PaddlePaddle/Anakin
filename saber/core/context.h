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

#ifndef ANAKIN_SABER_CORE_CONTEXT_H
#define ANAKIN_SABER_CORE_CONTEXT_H

#include "saber/core/env.h"
#include "saber/saber_types.h"
#ifdef USE_ARM_PLACE
#include "saber/core/tensor.h"
#endif

namespace anakin {
namespace saber {

template <typename TargetType>
class Context final {
    typedef TargetWrapper<TargetType> API;
public:
    typename Env<TargetType>::Devs& devs = Env<TargetType>::cur_env();
    /**
     * \brief context constructor, set device id, data stream id and compute stream id
     * @param device_id
     * @param data_stream_id
     * @param compute_stream_id
     */
    Context(int device_id = 0, int data_stream_id = 0, int compute_stream_id = 0) {
        CHECK_GT(devs.size(), 0) << "Env is not initialized or current target is not exit!";
        if (device_id >= devs.size()) {
            LOG(WARNING) << "device index exceeds the number of devices, set to default device(0)!";
            _device_id = 0;
        } else {
            _device_id = device_id;
        }
        if (devs[_device_id]._data_stream.size() < 1 || devs[_device_id]._compute_stream.size() < 1) {
            LOG(WARNING) << "device not initialized, create stream now";
            devs[_device_id].create_stream();
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
#ifdef USE_ARM_PLACE
        //! 1 thread, big core
        if (devs[_device_id]._info._big_core_ids.size() > 0){
            _act_ids = {devs[_device_id]._info._big_core_ids[0]};
        } else {
            _act_ids = {0};
        }
        _mode = SABER_POWER_HIGH;
        int temp_mem_size = devs[_device_id]._info._L2_cache[_act_ids[0]] / sizeof(float);
        _work_space.reshape(Shape({1, 1, 1, temp_mem_size}));
#ifdef TARGET_IOS
        _arch = APPLE; //use 6x8
#else
        if (devs[_device_id]._info._big_core_ids.size() > 0) {
            _arch = devs[_device_id]._info._archs[_act_ids[0]];
        }
#endif
#endif
    }

    Context(const Context<TargetType>& ctx) {

        _device_id = ctx._device_id;
        _data_stream_id = ctx._data_stream_id;
        _compute_stream_id = ctx._compute_stream_id;
        _stream_compute = ctx._stream_compute;
        _stream_data = ctx._stream_data;
#ifdef USE_ARM_PLACE
        _act_ids = ctx._act_ids;
        _mode = ctx._mode;
        _work_space.copy_from(ctx._work_space);
        _arch = ctx._arch;
        _count = ctx._count;
#endif
    }

    Context& operator=(const Context& ctx) {
        this->_device_id = ctx._device_id;
        this->_data_stream_id = ctx._data_stream_id;
        this->_compute_stream_id = ctx._compute_stream_id;
        this->_stream_data = ctx._stream_data;
        this->_stream_compute = ctx._stream_compute;
#ifdef USE_ARM_PLACE
        this->_act_ids = ctx._act_ids;
        this->_mode = ctx._mode;
        this->_work_space.copy_from(ctx._work_space);
        this->_arch = ctx._arch;
        this->_count = ctx._count;

#endif
        return *this;
    }

    ~Context() {
    }

    bool operator==(const Context &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_device_id == right._device_id);
        comp_eq = comp_eq && (_data_stream_id == right._data_stream_id);
        comp_eq = comp_eq && (_compute_stream_id == right._compute_stream_id);
#ifdef USE_ARM_PLACE
        comp_eq = comp_eq && (_act_ids == right._act_ids);
        comp_eq = comp_eq && (_mode == right._mode);
        comp_eq = comp_eq && (_arch == right._arch);
        comp_eq = comp_eq && (_count == right._count);
#endif

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
    typename API::stream_t get_data_stream() {
        return _stream_data;
    }

    /**
     * \brief get compute process stream
     * @return
     */
    typename API::stream_t get_compute_stream() {
        return _stream_compute;
    }

    std::string get_compute_ability() {
        // Fixme. need be !devs[_device_id]._info._compute_ability.empty()
        if (devs[_device_id]._compute_ability) {
            return devs[_device_id]._compute_ability;
        } else {
            return "null";
        }
    }
#ifdef USE_ARM_PLACE
    //! SABER_POWER_HIGH stands for using big cores,
    //! SABER_POWER_LOW stands for using small core,
    //! SABER_POWER_FULL stands for using all cores
    void set_run_mode(PowerMode mode, int threads);
    void set_cache(int l1size, int l2size, int l3size);
    int get_l1_cache_size() const;
    int get_l2_cache_size() const;
    int get_l3_cache_size() const;
    void* get_work_space();
    int get_threads() const;
    ARMArch get_arch() const;
    PowerMode get_mode() const;
    void set_arch(ARMArch arch);
    void bind_dev();
    SaberStatus workspace_extend(Shape sh);
#endif
    bool fusion() {return false;}

private:
    //! current stream to process
    typename API::stream_t _stream_data;
    typename API::stream_t _stream_compute;
    //! current device id
    int _device_id;
    int _data_stream_id;
    int _compute_stream_id;
#ifdef USE_ARM_PLACE
    ARMArch _arch;
    PowerMode _mode {SABER_POWER_HIGH};
    std::vector<int> _act_ids {0};
    Tensor<ARM> _work_space;
    long long _count {0};
#endif
};

} //namespace saber
} //namespace anakin

#ifdef USE_MLU
#include "saber/core/impl/mlu/mlu_context.h"
#endif

#ifdef USE_BM_PLACE
#include "saber/core/impl/bm/bm_context.h"
#endif


#endif //ANAKIN_SABER_CORE_CONTEXT_H
