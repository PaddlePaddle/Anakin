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

#include "core/env.h"
#include "saber/saber_types.h"
#include <type_traits>

#ifdef USE_BM
#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"
#endif

namespace anakin{

namespace saber{

template <typename TargetType>
class Context final{
    typedef TargetWrapper<TargetType> API;
public:
    typename Env<TargetType>::Devs& devs = Env<TargetType>::cur_env();
    /**
     * \brief context constructor, set device id, data stream id and compute stream id
     * @param device_id
     * @param data_stream_id
     * @param compute_stream_id
     */
    Context(int device_id = 0, int data_stream_id = 0, int compute_stream_id = 0){
#ifdef USE_BM        
        if(std::is_same<TargetType, BM>::value){
            LOG(INFO) << "context init for BM";
            _bm_handle = TargetWrapper<BM>::get_handle();
            return;
        }
#endif

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

    Context(const Context<TargetType>& ctx){
#ifdef USE_BM
        if(std::is_same<TargetType, BM>::value){
            LOG(INFO) << "context init for BM";
            _bm_handle = ctx._bm_handle;
            return;
        }
#endif
        _device_id = ctx._device_id;
        _data_stream_id = ctx._data_stream_id;
        _compute_stream_id = ctx._compute_stream_id;
        _stream_compute = ctx._stream_compute;
        _stream_data = ctx._stream_data;
#ifdef USE_ARM_PLACE
        _act_ids = ctx._act_ids;
        _mode = ctx._mode;
#endif

    }

    Context& operator=(const Context& ctx){
        this->_device_id = ctx._device_id;
        this->_data_stream_id = ctx._data_stream_id;
        this->_compute_stream_id = ctx._compute_stream_id;
        this->_stream_data = ctx._stream_data;
        this->_stream_compute = ctx._stream_compute;
#ifdef USE_ARM_PLACE
        this->_act_ids = ctx._act_ids;
        this->_mode = ctx._mode;
#endif
#ifdef USE_BM
        this->_bm_handle = ctx._bm_handle;
#endif
        return *this;
    }

    bool operator==(const Context &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_device_id == right._device_id);
        comp_eq = comp_eq && (_data_stream_id == right._data_stream_id);
        comp_eq = comp_eq && (_compute_stream_id == right._compute_stream_id);
#ifdef USE_BM
        comp_eq = comp_eq && (_bm_handle == right._bm_handle);
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
    typename API::stream_t get_data_stream(){
        return _stream_data;
    }

    /**
     * \brief get compute process stream
     * @return
     */
    typename API::stream_t get_compute_stream(){
        return _stream_compute;
    }


#ifdef USE_ARM_PLACE
    //void set_act_cores(std::vector<int> ids);
    //void set_power_mode(PowerMode mode);
    void set_run_mode(PowerMode mode, int threads);
    //void set_cache(size_t l1size, size_t l2size, size_t l3size);
    void bind_dev();
    PowerMode get_mode(int& threads);
    //PowerMode get_mode();
    //std::vector<int> get_act_ids();
#endif

#ifdef USE_BM
    bm_handle_t get_handle() {
        return _bm_handle;
    }
#endif


private:
    //! current stream to process
    typename API::stream_t _stream_data;
    typename API::stream_t _stream_compute;
    //! current device id
    int _device_id;
    int _data_stream_id;
    int _compute_stream_id;
#ifdef USE_ARM_PLACE
    PowerMode _mode{SABER_POWER_HIGH};
    std::vector<int> _act_ids{0};
#endif
#ifdef USE_BM
    bm_handle_t _bm_handle;
#endif
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_CONTEXT_H
