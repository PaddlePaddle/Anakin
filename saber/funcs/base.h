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

template<typename TargetType,
        DataType Dtype,
        template <typename T, DataType D, typename P> class Impl,
        template <typename T> class Param >
class BaseFunc {
public:
    typedef Param<TargetType> Param_t;
    typedef Impl<TargetType, Dtype, Param_t> Impl_t;
    typedef std::vector<Tensor<TargetType>*> Input_v;
    typedef std::vector<Tensor<TargetType>*> Output_v;
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
        Param_t& param, Context<TargetType> &ctx) {
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
              SaberImplStrategy strategy, ImplEnum implenum, Context<TargetType> &ctx) {

        this->_param = param;

        //this->_last_input_shape = input[0]->valid_shape();
        this->_last_input_shape.clear();
        for (int i = 0; i < input.size(); ++i) {
            this->_last_input_shape.push_back(input[i]->valid_shape());
        }
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
        Context<TargetType> &ctx) {

        bool last_shape_equal = false;
        last_shape_equal = input.size() == _last_input_shape.size();
        if (last_shape_equal) {
            for (int i = 0; i < input.size(); ++i) {
                last_shape_equal = last_shape_equal
                        && (_last_input_shape[i] == input[i]->valid_shape());
            }
        }    
        if ((_param == param) && last_shape_equal) {
            return _best_impl->dispatch(input, output, param);
        } else {
            _param = param;
//            this->_last_input_shape = input[0]->valid_shape();
            this->_last_input_shape.clear();
            for (int i = 0; i < input.size(); ++i) {
                this->_last_input_shape.push_back(input[i]->valid_shape());
            }
            reset_output_shape(input, output, param, ctx);
            pick_best(input, output, param, _strategy, _implenum, ctx);
            return _best_impl->dispatch(input, output, param);
        }
    }

protected:
    Param_t _param;
    Impl_t* _best_impl;
    std::vector<Shape> _last_input_shape;
    //std::unordered_map<Param_t, Impl_t*> _static_map;
    std::vector<Impl_t*> _impl;
    SaberImplStrategy _strategy;
    ImplEnum _implenum;

    void pick_best(const Input_v input, Output_v output, \
        Param_t& param, SaberImplStrategy strategy, ImplEnum implenum, \
        Context<TargetType> &ctx) {
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

    virtual void pick_best_runtime(const Input_v& input, Output_v& output, Param_t& param, \
        Context<TargetType> &ctx) {

        float time_cost = 99999.f;
        int idx = 0;

        std::vector<float> times;

        // warm up
        for (auto iter : _impl) {
            iter->dispatch(input, output, param);
        }

        for(auto iter : _impl) {
            SaberTimer<TargetType> timer;
            SaberStatus status = SaberUnImplError;
            for(int i = 0; i < _runtime_ts; ++i) {
                timer.start(ctx);
                status = SaberStatus(status | iter->dispatch(input, output, param));
                typename Tensor<TargetType>::API::stream_t stream = ctx.get_compute_stream();
                for (auto out : output) {
                    out->record_event(stream);
                    out->sync();
                }
                timer.end(ctx);
            }
            if (status == SaberSuccess) {
                times.push_back(timer.get_average_ms());
            } else {
                times.push_back(time_cost);
            }

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
