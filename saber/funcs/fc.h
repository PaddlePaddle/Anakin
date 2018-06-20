/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
   Modifications (c) 2018 Advanced Micro Devices, Inc.

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

#ifndef ANAKIN_SABER_FUNCS_FC_H
#define ANAKIN_SABER_FUNCS_FC_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_fc.h"
#include "saber/funcs/impl/cuda/vender_fc.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/vender_fc.h"
#endif

#ifdef USE_AMD
#include "saber/funcs/impl/amd/vender_fc.h"
#endif
   
namespace anakin{

namespace saber{

template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW
>
class Fc : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        FcParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            FcParam>::BaseFunc;

    Fc() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef FcParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {

        Shape shape_out = input[0]->valid_shape();
        int m = input[0]->count_valid(0, param.axis);
        int k = input[0]->count_valid(param.axis, input[0]->dims());
        int n = param.num_output;
        int weights_size = param.weights->valid_size();
        if (n <= 0) {
            n = weights_size / k;
        }
        CHECK_EQ(weights_size / n, k) << "weights size does not meet the input size";

        int num_idx = output[0]->num_index();
        int channel_idx = output[0]->channel_index();
        int height_idx = output[0]->height_index();
        int widht_idx = output[0]->width_index();
        if (num_idx >= 0) {
            shape_out[num_idx] = m;
        }
        if (height_idx >= 0) {
            shape_out[height_idx] = 1;
        }
        if (widht_idx >= 0) {
            shape_out[widht_idx] = 1;
        }
        shape_out[channel_idx] = n;
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderFc <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case SABER_IMPL:
                this->_impl.push_back(new SaberFc <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
#ifdef USE_CUDA
        bool use_saber_fc = true;
        use_saber_fc &= this->_last_input_shape[0] > 1;
        use_saber_fc &= this->_last_input_shape[0] <= 32;
        if (use_saber_fc) {
            this->_best_impl = this->_impl[1];
        } else {
            this->_best_impl = this->_impl[0];
        }
#endif
#ifdef USE_X86_PLACE
        this->_best_impl = this->_impl[0];
#endif
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_FC_H
