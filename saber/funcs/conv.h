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
#ifndef ANAKIN_SABER_FUNCS_CONV_H
#define ANAKIN_SABER_FUNCS_CONV_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/impl_conv.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_conv.h"
#endif

#ifdef USE_BM
#include "saber/funcs/impl/bm/vender_conv.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_conv.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class Conv : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ConvParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            ConvParam>::BaseFunc;

    Conv() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ConvParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        Shape conv_shape = conv_compute_shape(input[0]->valid_shape(), param);
        conv_shape.set_layout(Layout_NCHW);
        return output[0]->set_shape(conv_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderConv2D <TargetType,
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberConv2D <TargetType,
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