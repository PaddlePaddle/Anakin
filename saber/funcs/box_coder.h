/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_BOX_CODER_H
#define ANAKIN_SABER_FUNCS_BOX_CODER_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_box_coder.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_box_coder.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_box_coder.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
         DataType OpDtype>
class BoxCoder : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    BoxCoderParam > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    BoxCoderParam >::BaseFunc;

    BoxCoder() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef BoxCoderParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
            Output_v& output, Param_t& param) override {
        auto prior_box_tensor = input[0];
        auto loc_tensor = input[1];
        output[0]->set_seq_offset(loc_tensor->get_seq_offset());

        if (param.axis == 0) {
            CHECK_EQ(prior_box_tensor->num(), loc_tensor->channel());
        } else if (param.axis == 1) {
            CHECK_EQ(prior_box_tensor->num(), loc_tensor->num());
        } else {
            LOG(FATAL) << "invalid axis " << param.axis;
        }
        CHECK_EQ(prior_box_tensor->channel(), loc_tensor->width() + 1);
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(loc_tensor->valid_shape());

    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            this->_impl.push_back(new VenderBoxCoder <TargetType,
                                  OpDtype>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberBoxCoder <TargetType,
                                  OpDtype>);
            return SaberSuccess;

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

} // namespace saber
} // namespace anakin
#endif //ANAKIN_SABER_FUNCS_BOX_CODER_H
