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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_TOPK_AVG_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_TOPK_AVG_POOLING_H

#include "saber/funcs/impl/impl_topk_avg_pooling.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberTopKAvgPooling<NV, OpDtype> :
    public ImplBase<
        NV, OpDtype,
        TopKAvgPoolingParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    SaberTopKAvgPooling() = default;
    ~SaberTopKAvgPooling() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            TopKAvgPoolingParam<NV>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        int topk_num = param.top_ks.size();
        _top_ks.re_alloc(Shape({topk_num, 1, 1, 1}), AK_INT32);
        cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
        cudaMemcpyAsync(_top_ks.mutable_data(), &param.top_ks[0], sizeof(int) * topk_num, cudaMemcpyHostToDevice, cuda_stream);
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            TopKAvgPoolingParam<NV>& param, Context<NV> &ctx) {
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          TopKAvgPoolingParam<NV>& param);

protected:
   Tensor<NV> _height_offset;
   Tensor<NV> _width_offset;
   Tensor<NV> _top_ks;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_TOPK_POOLING_H
