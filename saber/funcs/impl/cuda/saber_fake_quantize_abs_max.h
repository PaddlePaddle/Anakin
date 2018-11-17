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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_FAKE_DQUANTIZE_ABS_MAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_FAKE_DQUANTIZE_ABS_MAX_H

#include "saber/funcs/impl/impl_fake_quantize_abs_max.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberFakeQuantizeAbsMax<NV, OpDtype>: public ImplBase<NV, OpDtype, FakeQuantizeAbsMaxParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberFakeQuantizeAbsMax() {}
    ~SaberFakeQuantizeAbsMax() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             FakeQuantizeAbsMaxParam<NV> &param,
                             Context<NV> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               FakeQuantizeAbsMaxParam<NV> &crop_param,
                               Context<NV> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 FakeQuantizeAbsMaxParam<NV> &param);

private:
    Tensor<NV> _max_abs;
    cudnnHandle_t _handle;
    cudnnReduceTensorDescriptor_t _reduce_tensor_descs;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    size_t _workspaceSizeInBytes;
    void *_workspace;
    size_t _indices_size;
    void *_indices;
    
};

template class SaberFakeQuantizeAbsMax<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_FAKE_QUANTIZE_ABS_MAX_H
