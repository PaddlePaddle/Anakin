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

#ifndef ANAKIN_SABER_FUNCS_MAXOUT_H
#define ANAKIN_SABER_FUNCS_MAXOUT_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_maxout.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_maxout.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_maxout.h"
#endif

#ifdef USE_AMD
#endif

#ifdef USE_ARM_PLACE
#endif

#ifdef USE_BM
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class MaxOut : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        MaxOutParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            MaxOutParam>::BaseFunc;

    MaxOut() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef MaxOutParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        
        CHECK_GE(param.groups, 1) << "MaxOut's groups must larger equal than 1.";
        CHECK_GE(input[0]->channel(), param.groups) << "[MaxOut] input channel must larger equal than groups.";
        int num = input[0]->num();
        int channel = input[0]->channel() / param.groups;
        int height = input[0]->height();
        int width = input[0]->width();

        Shape output_shape({num, channel, height, width});
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMaxOut <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberMaxOut <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace saber
} // namespace anakin

#endif
