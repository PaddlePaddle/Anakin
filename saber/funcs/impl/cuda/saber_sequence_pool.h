/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

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

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberSequencePool<NV, OpDtype, inDtype, outDtype,
          LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase <
          Tensor<NV, inDtype, LayOutType_in>,
          Tensor<NV, outDtype, LayOutType_out>,
          Tensor<NV, OpDtype, LayOutType_op>,
          SequencePoolParam<Tensor<NV, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    SaberSequencePool() = default;

    ~SaberSequencePool() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SequencePoolParam<OpTensor>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SequencePoolParam<OpTensor>& param,
                               Context<NV>& ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SequencePoolParam<OpTensor>& param) override;
private:
    typedef std::function<void(
            DataType_out* , const DataType_in* ,const int ,
    const int* , const int)> seq_pool_direct_kernel;
    std::map<SequencePoolType, seq_pool_direct_kernel> kernel_direct_map;

    Tensor<NV,AK_INT32,LayOutType_in> _seq_offset;

};
}
}

#endif