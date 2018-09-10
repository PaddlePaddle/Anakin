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

#ifndef ANAKIN_SABER_FUNCS_REVERSE_INPUT_H
#define ANAKIN_SABER_FUNCS_REVERSE_INPUT_H
#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_reverse_input.h"


#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_reverse_input.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_reverse_input.h"
#endif

#ifdef USE_AMD

#endif

#ifdef USE_ARM_PLACE

#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class ReverseInput : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        EmptyParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            EmptyParam>::BaseFunc;

    ReverseInput() = default;

    typedef Tensor <TargetType> InDataTensor;
    typedef Tensor <TargetType> OutDataTensor;
    typedef Tensor <TargetType> OpTensor;
    typedef EmptyParam <TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector <Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        for (int i = 0; i < input.size(); ++i) {
            output[i]->set_shape(input[i]->valid_shape());
            output[i]->set_seq_offset(input[i]->get_seq_offset());
        }
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderReverseInput <TargetType,
                OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberReverseInput <TargetType,
                OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true)
            this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

}
}
#endif //SABER_FUNCS_REVERSE_INPUT_H
