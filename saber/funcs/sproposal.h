/*
 * Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANAKIN_SABER_FUNCS_SPROPOSAL_H
#define ANAKIN_SABER_FUNCS_SPROPOSAL_H

#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_sproposal.h"

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_sproposal.h"
#endif
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_sproposal.h"
#endif

namespace anakin {
namespace saber {

template <typename TargetType,
        DataType OpDtype>
class SProposal : public BaseFunc <
        TargetType, OpDtype,
        ImplBase, SProposalParam
> {
public:
    typedef TargetType targetType_t;
    typedef Tensor<TargetType> OpTensor;
    typedef SProposalParam<TargetType> Param_t;
    typedef const std::vector<OpTensor*> Input_v;
    typedef std::vector<OpTensor*> Output_v;

    SProposal() = default;
    SaberStatus compute_output_shape(const Input_v &input,
                                     Output_v &output, Param_t &param) {

        // need to make sure the max size of this op.
        Shape output_shape({param.post_nms_topn, 5, 1, 1}, Layout_NCHW);
        return output[0]->set_shape_without_layout(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderSProposal<TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberSProposal<TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }

    };
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
#endif //ANAKIN_SABER_FUNCS_CONV_H
