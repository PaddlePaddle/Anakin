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

#ifndef ANAKIN_SABER_FUNCS_BASE_H
#define ANAKIN_SABER_FUNCS_BASE_H

#include "saber/saber_funcs_param.h"
#include "saber/core/context.h"
#include "timer.h"
#include <unordered_map>
#include <functional>

namespace anakin {

namespace saber {

template<typename inTensor, typename outTensor, typename opTensor,
    template <typename T0, typename T1, typename T2, typename T3> class Impl,
    template <typename T0> class Param
    >
class BaseFunc {
public:
    typedef typename inTensor::targetType_t targetType_t;
    typedef Param<opTensor> Param_t;
    typedef Impl<inTensor, outTensor, opTensor, Param_t> Impl_t;
    typedef std::vector<inTensor*> Input_v;
    typedef std::vector<outTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    BaseFunc() {}
    ~BaseFunc() {
        std::for_each(this->_impl.begin(), this->_impl.end(),
            [&](Impl_t* impl){
			if(impl) {
            	delete impl;
            	impl = nullptr;
			}
        });
    }

    // compute_output_shape()
    // return shape: (the layout is same with the input's)
    // for example: input(NCHW) -> return output_shape(NCHW)
    //              input(CHWN) -> return output_shape(CHWN)
    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
         Param_t& param) = 0;
    //TODO:create may lead to leak
    virtual SaberStatus reset_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param, Context<targetType_t> &ctx) {
        compute_output_shape(input, output, param);
        for (int i = 0; i < output.size(); ++i) {
            output[i]->reshape(output[i]->valid_shape());
        }
        for (auto imp : this->_impl) {
            if (imp) {
                SaberStatus status = imp->create(input, output, param, ctx);
                if (status != SaberSuccess) {
                    return status;
                }
            }
        }
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) = 0;

    virtual SaberStatus init(const Input_v& input, Output_v& output, Param_t& param,
              SaberImplStrategy strategy, ImplEnum implenum, Context<targetType_t > &ctx) {

        this->_param = param;
        this->_last_input_shape = input[0]->valid_shape();
        this->_strategy = strategy; 
		std::for_each(this->_impl.begin(), this->_impl.end(), 
				[&](Impl_t* impl){ 
					delete impl; 
					impl = nullptr; 
				}
		);

        this->_impl.clear();

        SaberStatus status = SaberSuccess;
        switch (strategy) {
            case RUNTIME:
                status = init_impl(VENDER_IMPL);
                status = SaberStatus(status | init_impl(SABER_IMPL));
                break;
            case SPECIFY:
                status = init_impl(implenum);
                break;
            case STATIC:
                status = init_impl(VENDER_IMPL);
                status = SaberStatus(status | init_impl(SABER_IMPL));
                break;
            default:
                status = SaberInvalidValue;
        }

        if (status != SaberSuccess) {
            return status;
        }

        for (auto imp : this->_impl) {
            status = SaberStatus(status | imp->init(input, output, param, ctx));
        }
        if (status != SaberSuccess) {
            return status;
        }

        this->pick_best(input, output, param, strategy, implenum, ctx);
        this->_param = param;
        return SaberSuccess;
    }

    virtual SaberStatus operator()(const Input_v& input, Output_v& output, Param_t& param, \
        Context<targetType_t> &ctx) {

        if ((_param == param) && (input[0]->valid_shape() == this->_last_input_shape)) {
            return _best_impl->dispatch(input, output, param);
        } else {
            _param = param;
            this->_last_input_shape = input[0]->valid_shape();
            reset_output_shape(input, output, param, ctx);
            pick_best(input, output, param, _strategy, _implenum, ctx);
            return _best_impl->dispatch(input, output, param);
        }
    }

protected:
    Param_t _param;
    Impl_t* _best_impl;
    Shape _last_input_shape;
    //std::unordered_map<Param_t, Impl_t*> _static_map;
    std::vector<Impl_t*> _impl;
    SaberImplStrategy _strategy;
    ImplEnum _implenum;

    void pick_best(const Input_v input, Output_v output, \
        Param_t& param, SaberImplStrategy strategy, ImplEnum implenum, \
        Context<targetType_t> &ctx) {
        switch(_strategy) {
            case STATIC:
                pick_best_static();
                break;
            case RUNTIME:
                pick_best_runtime(input, output, param, ctx);
                break;
            case SPECIFY:
                pick_best_specify(implenum);
                break;
            default:
                //err
                break;
        }
    }
private:
    const static int _runtime_ts = 10;

    //typedef std::unordered_map<Param_t, Impl*> static_map;
    virtual void pick_best_static() = 0;

    virtual void pick_best_runtime(const Input_v input, Output_v output, Param_t& param, \
        Context<targetType_t> &ctx) {

        float time_cost = 99999.f;
        int idx = 0;

        std::vector<float> times;

        // warm up
        for (auto iter : _impl) {
            iter->dispatch(input, output, param);
        }

        for(auto iter : _impl) {
            SaberTimer<targetType_t> timer;
            timer.start(ctx);
            for(int i = 0; i < _runtime_ts; ++i) {
                iter->dispatch(input, output, param);
            }
            output[0]->sync();
            timer.end(ctx);
            times.push_back(timer.get_average_ms());

        }
        for (int i = 0; i < _impl.size(); ++i) {

            if (time_cost > times[i]){
                time_cost = times[i];
                idx = i;
            }
        }
        _best_impl = _impl[idx];

    }

    virtual void pick_best_specify(ImplEnum implenum) = 0;

};


} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_BASE_H
