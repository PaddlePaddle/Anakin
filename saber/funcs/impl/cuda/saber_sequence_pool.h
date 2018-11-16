/* Copyright (c) 2018 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_POOL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_POOL_H

#include "saber/funcs/impl/impl_sequence_pool.h"
#include "saber/saber_funcs_param.h"
#include <functional>
#include <map>

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberSequencePool<NV, OpDtype> : 
    public ImplBase < NV, OpDtype, SequencePoolParam<NV> > {

public:
    typedef Tensor<NV> DataTensor_in;
    typedef Tensor<NV> DataTensor_out;
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<NV, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<NV, OpDtype>::Dtype DataType_op;

    SaberSequencePool() = default;

    ~SaberSequencePool() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SequencePoolParam<NV>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SequencePoolParam<NV>& param,
                               Context<NV>& ctx) {
	return SaberSuccess;

    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SequencePoolParam<NV>& param) override;
private:
    typedef std::function<void(DataType_out*, const DataType_in*, const int, \
            const int*, const int, Context<NV>* ctx)> seq_pool_direct_kernel;
    std::map<SequencePoolType, seq_pool_direct_kernel> kernel_direct_map;


};

}
}

#endif
