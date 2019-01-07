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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_FC_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_FC_H

#include "saber/funcs/impl/impl_fc.h"
#include "saber/funcs/gemm.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberFc<NV, OpDtype>: public ImplBase<NV, OpDtype, FcParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberFc() = default;
    ~SaberFc() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             FcParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               FcParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 FcParam<NV>& param);

private:
    Gemm<NV, SABER_IMPL, float, float> _gemm;
    bool _flag_trans_weights{false};
    int _M;
    int _K;
    int _N;
    bool _is_continue_buf{true};
};

} //namespace saber

} //namespace anakin

#endif
