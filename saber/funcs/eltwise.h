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

#ifndef ANAKIN_SABER_FUNCS_ELTWISE_H
#define ANAKIN_SABER_FUNCS_ELTWISE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_eltwise.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_eltwise.h"
#endif
#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_eltwise.h"
#endif
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_eltwise.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype
>
class Eltwise : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        EltwiseParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            EltwiseParam>::BaseFunc;

    Eltwise() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef EltwiseParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        for (int i = 1; i < input.size(); ++i) {
            CHECK_EQ(input[0]->num(), input[i]->num());
            CHECK_EQ(input[0]->channel(), input[i]->channel());
            CHECK_EQ(input[0]->height(), input[i]->height());
            CHECK_EQ(input[0]->width(), input[i]->width());
        }

        Shape output_shape = input[0]->valid_shape();
        output[0]->set_shape(output_shape);

        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderEltwise <TargetType,
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberEltwise <TargetType,
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


}

}

#endif //ANAKIN_SABER_FUNCS_ELTWISE_H