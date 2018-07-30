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
#ifndef ANAKIN_SABER_FUNCS_ARGMAX_H
#define ANAKIN_SABER_FUNCS_ARGMAX_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_argmax.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_argmax.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_argmax.h"
#endif

#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_argmax.h"
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
class Argmax : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ArgmaxParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ArgmaxParam>::BaseFunc;

    Argmax() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ArgmaxParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
                                             Output_v& output, Param_t& param) override {

        //! support inplace computation, output shape = input shape

        int num_top_axes = input[0]->dims();
        Shape output_shape = Shape::zero(num_top_axes);
        for (int i = 0; i < num_top_axes; ++i) {
            output_shape[i] = 1;
        }
        if (param.has_axis) {
            output_shape = input[0]->valid_shape();
            output_shape[param.axis] = param.top_k;
        } else {
            output_shape[0] = input[0]->valid_shape()[0];
            output_shape[2] = param.top_k;
            if (param.out_max_val) {
                output_shape[1] = 2;
            }
        }

        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderArgmax <TargetType,
                        OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberArgmax <TargetType,
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