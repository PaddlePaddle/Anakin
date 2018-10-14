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

#ifndef ANAKIN_SABER_FUNCS_POOLING_WITH_INDEX_H
#define ANAKIN_SABER_FUNCS_POOLING_WITH_INDEX_H

#include "saber/funcs/impl/impl_pooling_with_index.h"

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_pooling_with_index.h"
#endif
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_pooling_with_index.h"
#endif
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_pooling_with_index.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class PoolingWithIndex : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PoolingParam
> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PoolingParam>::BaseFunc;

    PoolingWithIndex() = default;

    typedef PoolingParam<TargetType> Param_t;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v &output, \
        Param_t& param) override {

        Shape output_shape = (input[0]->valid_shape());

        int in_height = input[0]->height();
        int in_width = input[0]->width();

        int window_h = param.window_h;
        int window_w = param.window_w;
        int pad_h = param.pad_h;
        int pad_w = param.pad_w;
        int stride_h = param.stride_h;
        int stride_w = param.stride_w;

        if (param.global_pooling){
            window_h = in_height;
            window_w = in_width;
        }

        int out_height = static_cast<int>(ceilf(static_cast<float>(
                             in_height + 2 * pad_h - window_h) / stride_h)) + 1;

        int out_width = static_cast<int>(ceilf(static_cast<float>(
                             in_width + 2 * pad_w - window_w) / stride_w)) + 1;

        if (param.pooling_padded()) {
            if ((out_height - 1) * stride_h >= in_height + pad_h) {
                -- out_height;
            }
            if ((out_width - 1) * stride_w >= in_width + pad_w) {
                -- out_width;
            }
        }

        int height_idx = input[0]->height_index();
        int width_idx = input[0]->width_index();

        output_shape[height_idx] = out_height;
        output_shape[width_idx] = out_width;
        output[1]->set_shape(output_shape);

        return output[0]->set_shape(output_shape);

    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderPoolingWithIndex <TargetType,
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberPoolingWithIndex <TargetType,
                        OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

}
}
#endif
