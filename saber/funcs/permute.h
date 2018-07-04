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

#ifndef ANAKIN_SABER_FUNCS_PERMUTE_H
#define ANAKIN_SABER_FUNCS_PERMUTE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_permute.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_permute.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_permute.h"
#endif
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_permute.h"
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
class Permute : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        PermuteParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            PermuteParam>::BaseFunc;

    Permute() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef PermuteParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {

        SaberStatus status;
        for (int i = 0; i < input.size(); ++i) {
            Shape output_shape = input[i]->valid_shape();

            if (input[i]->valid_shape().size() != param.order.size()) {
                LOG(FATAL) << "permute order param is not valid";
            }

            //for example: (n, h, w, c)->(n, c, h, w)  by order(0, 3, 1, 2)
            for (int j = 0; j < param.order.size(); j++) {
                output_shape[j] = input[i]->valid_shape()[param.order[j]];
            }
            output[i]->set_shape(output_shape);
        }
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderPermute <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberPermute <TargetType, OpDtype, inDtype, outDtype,
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

    //virtual void pick_best_runtime(Input_v input, Output_v output, Param_t& param) override {}

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

}
}

#endif //ANAKIN_SABER_FUNCS_PERMUTE_H
