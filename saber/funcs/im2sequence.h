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

#ifndef ANAKIN_SABER_FUNCS_IM2SEQUENCE_H
#define ANAKIN_SABER_FUNCS_IM2SEQUENCE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_im2sequence.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_im2sequence.h"
#endif
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_im2sequence.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_im2sequence.h"
#endif
#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_im2sequence.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType, DataType OpDtype>
class Im2Sequence : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        Im2SequenceParam> {
public:
    using BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        Im2SequenceParam>::BaseFunc;

    Im2Sequence() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef Im2SequenceParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        SaberStatus status;
        Shape output_shape = input[0]->valid_shape();
        if (input[0]->shape().size() < 4) {
            LOG(FATAL) << "using reshape2d to reshape a 1d conv?";
        }
        int num_idx = input[0]->num_index();
        int channel_idx = input[0]->channel_index();
        int height_idx = input[0]->height_index();
        int width_idx = input[0]->width_index();
        int input_height = input[0]->height(); // P
        int kernel_exten_h = param.dilation_h * (param.window_h - 1) + 1;
        int output_height = (input_height + param.pad_up + param.pad_down - kernel_exten_h)
                         / param.stride_h + 1;

        int input_width = input[0]->width(); // Q
        int kernel_exten_w = param.dilation_w * (param.window_w - 1) + 1;
        int output_width = (input_width + param.pad_left + param.pad_right - kernel_exten_w)
                     / param.stride_w + 1;

        output_shape[num_idx] = input[0]->num() * output_height * output_width; // N
        output_shape[channel_idx] = input[0]->channel() * param.window_h * param.window_w; // K
        output_shape[height_idx] = 1;
        output_shape[width_idx] = 1;
        output[0]->set_shape(output_shape);

        int n=input[0]->num();
        std::vector<int> offset0(n+1);
        std::vector<std::vector<int>> offset;
        offset.push_back(offset0);
        for(int i=0;i<=n;i++){
            offset[0].push_back(i*output_height * output_width);
        }
        output[0]->set_seq_offset(offset);
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderIm2Sequence <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberIm2Sequence <TargetType, OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_IM2SEQUENCE_H
