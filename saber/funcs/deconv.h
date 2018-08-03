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
#ifndef ANAKIN_SABER_FUNCS_DECONV_H
#define ANAKIN_SABER_FUNCS_DECONV_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_deconv.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_deconv.h"
#include "saber/funcs/impl/cuda/vender_deconv.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_deconv.h"
#endif
#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/arm/saber_deconv.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW
>
class Deconv : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ConvParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ConvParam>::BaseFunc;

    Deconv() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ConvParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input, \
        Output_v &output, Param_t &param) override {
        Shape output_shape = (input[0]->shape());

        if (input[0]->shape().size() < 4) {
                    LOG(FATAL) << "using reshape2d to reshape a 1d conv?";
        }

        // append the $n and $c/$k, output: N * K * P * Q
        int num_idx = input[0]->num_index();
        int channel_idx = input[0]->channel_index();
        int height_idx = input[0]->height_index();
        int width_idx = input[0]->width_index();

        output_shape[num_idx] = input[0]->num(); // N
        output_shape[channel_idx] = param.weight()->num() * param.group; // K

        int kernel_extent_h = param.dilation_h *
                                      (param.weight()->height() - 1) + 1;
        int output_dim_h = (input[0]->height() - 1) *
                                   param.stride_h + kernel_extent_h - 2 * param.pad_h;
        int kernel_extent_w = param.dilation_w *
                                      (param.weight()->width() - 1) + 1;
        int output_dim_w = (input[0]->width() - 1) *
                                   param.stride_w + kernel_extent_w - 2 * param.pad_w;

        output_shape[height_idx] = output_dim_h;
        output_shape[width_idx] = output_dim_w;
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderDeconv2D <TargetType,
                        OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberDeconv2D <TargetType,
                        OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
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

} // namespace saber
} // namespace anakin


#endif