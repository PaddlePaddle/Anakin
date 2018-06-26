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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_GRU_H

#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_gru.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/impl/x86/x86_utils.h"


#include <mkl_cblas.h>
#include <mkl_lapacke.h>

namespace anakin {
namespace saber {

template<DataType OpDtype,
         DataType inDtype,
         DataType outDtype,
         typename LayOutType_op,
         typename LayOutType_in,
         typename LayOutType_out>

class VenderGru<X86, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out> :
    public ImplBase <
    Tensor<X86, inDtype, LayOutType_in>,
    Tensor<X86, outDtype, LayOutType_out>,
    Tensor<X86, OpDtype, LayOutType_op>,
    GruParam<Tensor<X86, OpDtype, LayOutType_op> >> {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderGru() : avx2_available_(false), aligned_bias_(nullptr),
                  max_thread_num_(1),
        weight_x_packed_(nullptr),
        weight_ru_packed_(nullptr),
        weight_c_packed_(nullptr) {
        LOG(INFO)<<"init vender gru";
    }

    ~VenderGru() {
        if (this->weight_x_packed_) {
            cblas_sgemm_free(this->weight_x_packed_);
            this->weight_x_packed_ = nullptr;
        }

        if (this->weight_ru_packed_) {
            cblas_sgemm_free(this->weight_ru_packed_);
            this->weight_ru_packed_ = nullptr;
        }

        if (this->weight_c_packed_) {
            cblas_sgemm_free(this->weight_c_packed_);
            this->weight_c_packed_ = nullptr;
        }

        if (this->aligned_bias_) {
            zfree(this->aligned_bias_);
            this->aligned_bias_ = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             GruParam<OpTensor>& gru_param,
                             Context<X86>& ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               GruParam<OpTensor>& gru_param,
                               Context<X86>& ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 GruParam<OpTensor>& param) override;

private:
    bool avx2_available_;
    int max_thread_num_;
    int word_size_;
    int hidden_size_;
    int aligned_hidden_size_;

    float* aligned_bias_;
    OpTensor aligned_init_hidden;

    OpDataType* weight_x_packed_ = nullptr;
    OpDataType* weight_ru_packed_ = nullptr;
    OpDataType* weight_c_packed_ = nullptr;
    DataTensor_out batched_h;
    DataTensor_in batched_x;
    DataTensor_out batched_xx;

    SaberStatus check_conf(const std::vector<DataTensor_in*>& inputs,
                           std::vector<DataTensor_out*>& outputs,
                           GruParam<OpTensor>& param);
};

} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_GRU_H
