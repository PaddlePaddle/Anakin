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

#ifndef ANAKIN_SABER_FUNCS_ANCHOR_GENERATOR_H
#define ANAKIN_SABER_FUNCS_ANCHOR_GENERATOR_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_anchor_generator.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_anchor_generator.h"
//#include "saber/funcs/impl/cuda/vender_anchor_generator.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_anchor_generator.h"
#endif
#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_anchor_generator.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType, DataType OpDtype>
class AnchorGenerator : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        AnchorGeneratorParam> {
public:
    using BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        AnchorGeneratorParam>::BaseFunc;

    AnchorGenerator() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef AnchorGeneratorParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        SaberStatus status;
        CHECK_EQ(input.size(), 1);
        CHECK_EQ(output.size(), 2);
        auto anchor_sizes = param.anchor_sizes;
        auto aspect_ratios = param.aspect_ratios;
        int num_anchors = anchor_sizes.size() * aspect_ratios.size();
        Shape output_shape = std::vector<int>{input[0]->height(), input[0]->width(), num_anchors, 4};
        output[0]->set_shape(output_shape);
        output[1]->set_shape(output_shape);

        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderAnchorGenerator <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberAnchorGenerator <TargetType, OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_ANCHOR_GENERATOR_H
