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

#ifndef ANAKIN_SABER_FUNCS_TOPK_AVG_POOLING_H
#define ANAKIN_SABER_FUNCS_TOPK_AVG_POOLING_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_topk_avg_pooling.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_topk_avg_pooling.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_topk_avg_pooling.h"
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
class TopKAvgPooling : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        TopKAvgPoolingParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            TopKAvgPoolingParam>::BaseFunc;

    TopKAvgPooling() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef TopKAvgPoolingParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        int num_k = param.top_ks.size();
        int dim0 = -1;
        if (param.is_pooling_by_row) {
            dim0 = input[1]->num();
            output[0]->set_seq_offset(input[1]->get_seq_offset());
        } else {
            dim0 = input[2]->num();
            output[0]->set_seq_offset(input[1]->get_seq_offset());
        }
        auto offset = output[0]->get_seq_offset()[0];
        Shape output_shape({offset[offset.size() - 1], param.feat_map_num * num_k, 1, 1});
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                //this->_impl.push_back(new VenderTopKAvgPooling <TargetType,
                this->_impl.push_back(new VenderTopKAvgPooling <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberTopKAvgPooling <TargetType, OpDtype>);
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
