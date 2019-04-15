/* Copyright (c) 2018 Anakin Authors All Rights Reserve.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_VENDER_FC_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_VENDER_FC_H

#include <vector>

#include "mkl_cblas.h"
#include "saber/funcs/impl/impl_fc.h"
#include "saber/funcs/impl/x86/mkl_packed_int8_gemm.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class VenderFc<X86, OpDtype> : public ImplBase<X86, OpDtype, FcParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    VenderFc() : bias_sum(nullptr),_need_weights_trans(false),ws_(nullptr),MB(0),OC(0),
                 _batch_size(0),_output_channel(0),_is_transpose_weights(CblasNoTrans)
    {}

    ~VenderFc() {
        clean();
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             FcParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               FcParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 FcParam<X86> &param) override;
    virtual void clean();

private:
    OpDataType *bias_sum;
    int MB;
    int OC;
    Tensor<X86> _weights_trans;
    bool _need_weights_trans;
    std::vector<float*> packed_weights;
    void *ws_;
    int _batch_size;
    int _output_channel;
    std::vector<float> _scale;
    CBLAS_TRANSPOSE _is_transpose_weights;//trans in mklml
    Tensor<X86> _input_scale;
    Tensor<X86> _bias_scale;

    PackedMKLInt8Gemm _packed_int8_gemm;
};


} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_VENDER_FC_H
