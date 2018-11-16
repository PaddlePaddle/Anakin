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

template <DataType OpDtype>
class VenderFc<X86, OpDtype> : public ImplBase<X86, OpDtype, FcParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    VenderFc() : bias_sum(nullptr)
    {}

    ~VenderFc() {
        if (bias_sum) {
            free(bias_sum);
            bias_sum = nullptr;
        }

        for (int i = packed_weights.size() - 1; i >= 0; i--) {
           OpDataType *pw = packed_weights[i];
           cblas_sgemm_free(pw);
           pw = nullptr;
           packed_weights.pop_back();
        }
        std::vector<OpDataType*> ().swap(packed_weights);
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

private:
    OpDataType *bias_sum;
    int MB;
    int OC;
    std::vector<OpDataType*> packed_weights;
};


} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_VENDER_FC_H
