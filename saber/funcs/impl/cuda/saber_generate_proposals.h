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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GENERATE_PROPOSALS_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GENERATE_PROPOSALS_H

#include "saber/funcs/impl/impl_generate_proposals.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberGenerateProposals<NV, OpDtype> :
    public ImplBase<
        NV, OpDtype,
        GenerateProposalsParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    SaberGenerateProposals() = default;
    ~SaberGenerateProposals() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            GenerateProposalsParam<NV>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        auto scores = inputs[0];
        auto bbox_deltas = inputs[1];
        Shape scores_shape = scores->valid_shape();
        Shape scores_swap_shape({scores_shape[0],
                                 scores_shape[2],
                                 scores_shape[3],
                                 scores_shape[1]}, Layout_NCHW);

        Shape bbox_deltas_shape = bbox_deltas->valid_shape();
        Shape bbox_deltas_swap_shape({bbox_deltas_shape[0],
                                     bbox_deltas_shape[2],
                                     bbox_deltas_shape[3],
                                     bbox_deltas_shape[1]}, Layout_NCHW);
        _scores_swap.reshape(scores_swap_shape);
        _bbox_deltas_swap.reshape(bbox_deltas_swap_shape);
        _scores_index.reshape(inputs[0]->valid_shape());
        _sorted_scores.reshape(inputs[0]->valid_shape());
        _sorted_index.reshape(inputs[0]->valid_shape());
        _sorted_index.set_dtype(AK_INT32);

        int batch_size = inputs[0]->num();
        _proposals.reshape(std::vector<int>{batch_size, param.pre_nms_top_n, 4, 1});
        _keep_num.reshape(std::vector<int>{batch_size, 1, 1, 1});
        _keep_num.set_dtype(AK_INT32);
        _keep.reshape(std::vector<int>{batch_size, param.pre_nms_top_n, 1, 1});
        _keep.set_dtype(AK_INT32);
        _keep_nms.reshape(std::vector<int>{1, param.pre_nms_top_n, 1, 1});
        _boxes_out.reshape(std::vector<int>{param.pre_nms_top_n, 5, 1, 1});
        _scores_out.reshape(std::vector<int>{param.pre_nms_top_n, 1, 1, 1});
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            GenerateProposalsParam<NV>& param, Context<NV> &ctx) {
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          GenerateProposalsParam<NV>& param);
private:
    Tensor<NV> _scores_swap;
    Tensor<NV> _bbox_deltas_swap;
    Tensor<NV> _scores_index;
    Tensor<NV> _sorted_scores;
    Tensor<NV> _sorted_index;
    Tensor<NV> _proposals;
    Tensor<NV> _keep_num;
    Tensor<NV> _keep;
    Tensor<NV> _keep_nms;
    Tensor<NV> _boxes_out;
    Tensor<NV> _scores_out;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GENERATE_PROPOSALS_H
