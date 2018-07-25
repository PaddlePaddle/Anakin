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
#ifndef ANAKIN_SABER_FUNCS_DECONV_ACT_H
#define ANAKIN_SABER_FUNCS_DECONV_ACT_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_deconv_act.h"
#include "saber/funcs/funcs_utils.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_deconv_act.h"
#include "saber/funcs/impl/cuda/vender_deconv_act.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_deconv_act.h"
#endif
#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/arm/saber_deconv_act.h"
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
class DeconvAct : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ConvActiveParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ConvActiveParam>::BaseFunc;

    DeconvAct() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ConvActiveParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
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
        ConvParam<OpTensor> conv_param = param.conv_param;
        output_shape[channel_idx] = conv_param.weight()->num() * conv_param.group; // K

        int kernel_extent_h = conv_param.dilation_h *
                                      (conv_param.weight()->height() - 1) + 1;
        int output_dim_h = (input[0]->height() - 1) *
                                   conv_param.stride_h + kernel_extent_h - 2 * conv_param.pad_h;
        int kernel_extent_w = conv_param.dilation_w *
                                      (conv_param.weight()->width() - 1) + 1;
        int output_dim_w = (input[0]->width() - 1) *
                                   conv_param.stride_w + kernel_extent_w - 2 * conv_param.pad_w;

        output_shape[height_idx] = output_dim_h;
        output_shape[width_idx] = output_dim_w;
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderDeconv2DAct <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberDeconv2DAct <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }
virtual SaberStatus init(const Input_v& input, Output_v& output, Param_t& param,
                      SaberImplStrategy strategy, ImplEnum implenum,
                      Context<TargetType> &ctx) override {

        update_weights(param);

        return BaseFunc<Tensor<TargetType, inDtype, LayOutType_in>,
                Tensor<TargetType, outDtype, LayOutType_out>,
                Tensor<TargetType, OpDtype, LayOutType_op>,
                ImplBase,
                ConvActiveParam>::init(input, output, param, strategy, implenum, ctx);
    }

    //should move this funcs to utils
    void update_weights(ConvActiveParam<OpTensor> &param) {
        update_deconv_weights<OpTensor, ConvActiveParam>(param);
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
