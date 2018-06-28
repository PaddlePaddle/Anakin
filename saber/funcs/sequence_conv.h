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

#ifndef ANAKIN_SABER_FUNCS_SEQUENCE_CONV_H
#define ANAKIN_SABER_FUNCS_SEQUENCE_CONV_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"

#ifdef NVIDIA_GPU
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_sequence_conv.h"
#endif

#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_sequence_conv.h"
#endif

#ifdef USE_AMD
//todo
#include "saber/funcs/impl/impl_sequence_conv.h"
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
class SequenceConv : public BaseFunc <
    Tensor<TargetType, inDtype, LayOutType_in>,
    Tensor<TargetType, outDtype, LayOutType_out>,
    Tensor<TargetType, OpDtype, LayOutType_op>,
    ImplBase,
    SequenceConvParam
    > {
public:
    using BaseFunc <
    Tensor<TargetType, inDtype, LayOutType_in>,
           Tensor<TargetType, outDtype, LayOutType_out>,
           Tensor<TargetType, OpDtype, LayOutType_op>,
           ImplBase,
           SequenceConvParam >::BaseFunc;

    SequenceConv() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef SequenceConvParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
            Output_v& output, Param_t& param) override {
        InDataTensor* input_tensor = input[0];
        Shape new_shape(input_tensor->num(), param.filter_tensor->width(), 1, 1);
        return output[0]->set_shape(new_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
#ifdef USE_X86_PLACE
        case VENDER_IMPL:
            CHECK_EQ(1, 0) << "Sequence conv No Vender imp";
            //                        this->_impl.push_back(new VenderSequencePool <TargetType, OpDtype, inDtype, outDtype,
            //                                LayOutType_op, LayOutType_in, LayOutType_out>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberSequenceConv <TargetType, OpDtype, inDtype, outDtype,
                                  LayOutType_op, LayOutType_in, LayOutType_out>);
            return SaberSuccess;
#endif

        default:
            return SaberUnImplError;
        }
    }
private:

    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};
}
}


#endif
