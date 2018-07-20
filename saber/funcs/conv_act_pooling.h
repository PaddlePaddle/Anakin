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
#ifndef ANAKIN_SABER_FUNCS_CONV_ACT_POOLING_H
#define ANAKIN_SABER_FUNCS_CONV_ACT_POOLING_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/impl_conv_act_pooling.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_conv_act_pooling.h"
#include "saber/funcs/impl/cuda/vender_conv_act_pooling.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_conv_act_pooling.h"
#endif

#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/arm/saber_conv_act_pooling.h"
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
class ConvActPooling : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ConvActivePoolingParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ConvActivePoolingParam>::BaseFunc;

    ConvActPooling() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ConvActivePoolingParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {


        Shape conv_shape = (input[0]->valid_shape());

        if (input[0]->valid_shape().size() < 4) {
            return SaberInvalidValue;
        }

        // append the $n and $c/$k, output: N * K * P * Q
        int num_idx = input[0]->num_index();
        int channel_idx = input[0]->channel_index();
        int height_idx = input[0]->height_index();
        int width_idx = input[0]->width_index();

        conv_shape[num_idx] = input[0]->num(); // N
        conv_shape[channel_idx] = param.conv_param.weight()->num(); // K

        int input_dim = input[0]->height(); // P
        int kernel_exten = param.conv_param.dilation_h *
                           (param.conv_param.weight()->height() - 1) + 1;
        int output_dim = (input_dim + 2 * param.conv_param.pad_h - kernel_exten)
                         / param.conv_param.stride_h + 1;

        conv_shape[height_idx] = output_dim;

        input_dim = input[0]->width(); // Q
        kernel_exten = param.conv_param.dilation_w * (param.conv_param.weight()->width() - 1) + 1;
        output_dim = (input_dim + 2 * param.conv_param.pad_w - kernel_exten)
                     / param.conv_param.stride_w + 1;

        conv_shape[width_idx] = output_dim;

        _conv_shape = conv_shape;
        Shape output_shape = conv_shape;

        int in_height = conv_shape[height_idx];
        int in_width = conv_shape[width_idx];

        int window_h = param.pooling_param.window_h;
        int window_w = param.pooling_param.window_w;
        int pad_h = param.pooling_param.pad_h;
        int pad_w = param.pooling_param.pad_w;
        int stride_h = param.pooling_param.stride_h;
        int stride_w = param.pooling_param.stride_w;
        int out_height;
        int out_width;
        if (param.pooling_param.global_pooling) {
            out_height = 1;
            out_width = 1;
            param.pooling_param.stride_h = in_height;
            param.pooling_param.stride_w = in_width;
            window_h = in_height;
            window_w = in_width;
            param.pooling_param.window_h = in_height;
            param.pooling_param.window_w = in_width;
        } else {
            if (param.pooling_param.cmp_out_shape_floor_as_conv) {
                out_height = static_cast<int>((static_cast<float>(
                                                       in_height + 2 * pad_h - window_h) / stride_h)) + 1;

                out_width = static_cast<int>((static_cast<float>(
                                                      in_width + 2 * pad_w - window_w) / stride_w)) + 1;
            } else {
                out_height = static_cast<int>(ceilf(static_cast<float>(
                                                            in_height + 2 * pad_h - window_h) / stride_h)) + 1;

                out_width = static_cast<int>(ceilf(static_cast<float>(
                                                           in_width + 2 * pad_w - window_w) / stride_w)) + 1;
            }
        }

        if (param.pooling_param.pooling_padded()) {
            if ((out_height - 1) * stride_h >= in_height + pad_h) {
                -- out_height;
            }
            if ((out_width - 1) * stride_w >= in_width + pad_w) {
                -- out_width;
            }
        }

        output_shape[height_idx] = out_height;
        output_shape[width_idx] = out_width;

        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderConv2DActPooling <TargetType,
                        OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberConv2DActPooling <TargetType,
                        OpDtype, inDtype, outDtype,
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
                ConvActivePoolingParam>::init(input, output, param, strategy, implenum, ctx);
    }

    //should move this funcs to utils
    void update_weights(ConvActivePoolingParam<OpTensor> &param) {
        update_conv_weights<OpTensor, ConvActivePoolingParam>(param);
    }
private:

    virtual void pick_best_static() override {

        bool _use_saber_conv_pooling = true;
        _use_saber_conv_pooling &= (this->_param).pooling_param.pad_h == 0;
        _use_saber_conv_pooling &= (this->_param).pooling_param.pad_w == 0;
        _use_saber_conv_pooling &= (this->_param).pooling_param.stride_h == 2;
        _use_saber_conv_pooling &= (this->_param).pooling_param.stride_w == 2;
        _use_saber_conv_pooling &= (this->_param).pooling_param.window_h == 2;
        _use_saber_conv_pooling &= (this->_param).pooling_param.window_w == 2;
        _use_saber_conv_pooling &= !(this->_param).pooling_param.global_pooling;
        _use_saber_conv_pooling &= (this->_param).pooling_param.pooling_type == Pooling_max;
	_use_saber_conv_pooling &= ((this->_last_input_shape[2] % 2) == 0);
        _use_saber_conv_pooling &= ((this->_last_input_shape[3] % 2) == 0);

        if (_use_saber_conv_pooling) {
            this->_best_impl = this->_impl[1];
            delete this->_impl[0];
            this->_impl[0] = NULL;
        } else {
            this->_best_impl = this->_impl[0];
            delete this->_impl[1];
            this->_impl[1] = NULL;
        }
    }

    //virtual void pick_best_runtime(Input_v input, Output_v output, Param_t& param) override {}

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
    Shape _conv_shape;
};

} // namespace saber
} // namespace anakin


#endif
