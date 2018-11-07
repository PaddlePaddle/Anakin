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
#include "saber/funcs/impl/x86/saber_argmax.h"
#endif

#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_argmax.h"
#endif

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_argmax.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class Argmax : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ArgmaxParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            ArgmaxParam>::BaseFunc;

    Argmax() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ArgmaxParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
                                             Output_v& output, Param_t& param) override {

        //! support inplace computation, output shape = input shape

        int top_k = param.top_k;
        bool out_max_val = param.out_max_val;
        bool has_axis = param.has_axis;
        int axis = param.axis;
        CHECK_GE(top_k, 1) << "top k must not less than 1.";
        if(has_axis){
           CHECK_GE(axis, 0) << "axis must not less than 0.";
           CHECK_LE(axis, input[0]->dims()) << "axis must be less than or equal to the number od dims.";
           CHECK_LE(top_k, input[0]->valid_shape()[axis]) << "top_k must be less than or equal to the dimension of the axis.";
        } else{
           CHECK_LE(top_k, input[0]->count(1, input[0]->dims())) << "top_k must be less than or equal to the dimension of input.";
        }
        //int num_top_axes = input[0]->dims();
       // if(num_top_axes < 3) num_top_axes = 3;
        Shape output_shape({1, 1, 1, 1}, Layout_NCHW);
        //Shape output_shape = Shape::zero(num_top_axes);
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
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberArgmax <TargetType,
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

} // namespace saber
} // namespace anakin


#endif
