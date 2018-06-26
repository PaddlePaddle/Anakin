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

namespace anakin {
namespace saber {

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase<
        Tensor<X86, inDtype, LayOutType_in>,
        Tensor<X86, outDtype, LayOutType_out>,
        Tensor<X86, OpDtype, LayOutType_op>,
        FcParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    VenderFc() : bias_sum(nullptr)
    {}

    ~VenderFc() {
        if (bias_sum) {
            free(bias_sum);
            bias_sum = nullptr;
        }

        for (int i = packed_weights.size() - 1; i >= 0; i--) {
           DataType_op *pw = packed_weights[i];
           cblas_sgemm_free(pw);
           pw = nullptr;
           packed_weights.pop_back();
        }
        std::vector<DataType_op*> ().swap(packed_weights);
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             FcParam<OpTensor> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               FcParam<OpTensor> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 FcParam<OpTensor> &param) override;

private:
    DataType_op *bias_sum;
    int MB;
    int OC;
    std::vector<DataType_op*> packed_weights;
};


} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_VENDER_FC_H
