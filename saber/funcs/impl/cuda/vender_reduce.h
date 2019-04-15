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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_REDUCE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_REDUCE_H

#include "saber/funcs/impl/impl_reduce.h"
#include <functional>
#include <map>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderReduce<NV, OpDtype> :
        public ImplBase<
                NV, OpDtype,
                ReduceParam<NV> > {
public:
    VenderReduce() = default;
    ~VenderReduce() {
        CUDNN_CHECK(cudnnDestroy(_handle));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(_reduce_descs));
        cudaFree(_workspace);
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ReduceParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ReduceParam<NV>& param, Context<NV> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
            std::vector<Tensor<NV>*>& outputs,
            ReduceParam<NV>& param);

private:
    cudnnHandle_t _handle{nullptr};
    cudnnTensorDescriptor_t _input_descs{nullptr};
    cudnnTensorDescriptor_t _output_descs{nullptr};
    cudnnReduceTensorDescriptor_t _reduce_descs{nullptr};
    size_t _workspace_fwd_sizes{0};
    void *_workspace{nullptr};  // aliases into _workspaceData
};
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_REDUCE_H
