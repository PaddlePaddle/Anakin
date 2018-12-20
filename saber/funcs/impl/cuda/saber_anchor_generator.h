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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ANCHOR_GENERATOR_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ANCHOR_GENERATOR_H

#include "saber/funcs/impl/impl_anchor_generator.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberAnchorGenerator<NV, OpDtype>: public ImplBase<NV, OpDtype, AnchorGeneratorParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberAnchorGenerator() {}
    ~SaberAnchorGenerator() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             AnchorGeneratorParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               AnchorGeneratorParam<NV> &param,
                               Context<NV> &ctx) {
        Shape shape_aspect({1, (int)(param.aspect_ratios.size()), 1, 1}, Layout_NCHW);
        Shape shape_anchor_sizes({1, (int)(param.anchor_sizes.size()), 1, 1}, Layout_NCHW);
        _aspect_ratios.reshape(shape_aspect);
        _anchor_sizes.reshape(shape_anchor_sizes);
        cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
        cudaMemcpyAsync((float*)(_aspect_ratios.mutable_data()), 
                &param.aspect_ratios[0], 
                sizeof(float) * param.aspect_ratios.size(), 
                cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync((float*)(_anchor_sizes.mutable_data()), 
                &param.anchor_sizes[0], 
                sizeof(float) * param.anchor_sizes.size(),
                cudaMemcpyHostToDevice, 
                cuda_stream);
        CHECK_EQ(param.stride.size(), 2) << "anchor generator stride size must be equal to 2";
        CHECK_EQ(param.variances.size(), 4) << "anchor generator variances size must be equal to 4";
        
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 AnchorGeneratorParam<NV> &param);

private:
    Tensor<NV> _aspect_ratios;
    Tensor<NV> _anchor_sizes;
};

template class SaberAnchorGenerator<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ANCHOR_GENERATOR_H
