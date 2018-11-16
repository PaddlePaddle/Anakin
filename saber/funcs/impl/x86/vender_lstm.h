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
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_LSTM_H

#include "saber/funcs/impl/impl_lstm.h"
#include "saber_funcs_param.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <x86intrin.h>
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "mkl_service.h"
namespace anakin {
namespace saber {
template<DataType OpDtype>
class VenderLstm<X86, OpDtype>: public ImplBase <
    X86, OpDtype, LstmParam<X86 >> {
public:
    typedef Tensor<X86> OpTensor;
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    VenderLstm() : avx2_available_(false),
        max_thread_num_(1),
        aligned_bias_(nullptr),
        aligned_init_hidden_(nullptr) {}

    ~VenderLstm() {
        if (weight_x_packed_.size()) {
            for (int i = 0; i < weight_x_packed_.size(); i++) {
                cblas_sgemm_free(weight_x_packed_[i]);
            }
        }

        if (weight_h_packed_.size()) {
            for (int i = 0; i < weight_h_packed_.size(); i++) {
                cblas_sgemm_free(weight_h_packed_[i]);
            }
        }

        if (this->aligned_bias_) {
            zfree(this->aligned_bias_);
            this->aligned_bias_ = nullptr;
        }

        if (this->aligned_init_hidden_) {
            zfree(this->aligned_init_hidden_);
            this->aligned_init_hidden_ = nullptr;
        }
    }
    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             LstmParam<X86>& param,
                             Context<X86>& ctx) override;
    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               LstmParam<X86>& param,
                               Context<X86>& ctx) override;
    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 LstmParam<X86>& param) override;
private:
    bool avx2_available_;
    int max_thread_num_;
    int word_size_;
    int hidden_size_;
    int aligned_hidden_size_;
    bool create_done = false;
    int direction_parallel_num_ = 2;
    int wave_front_thread_num_ = 1;
    int mkl_thread_num_ = 1;
    OpDataType* aligned_bias_;
    OpDataType* aligned_init_hidden_;
    OpDataType* aligned_init_hidden_c;
    std::vector<float *> weight_x_packed_;
    std::vector<float*> weight_h_packed_;
    std::vector<float*> aligned_wx_;
    std::vector<float*> aligned_wh_;
    OpTensor batched_h;
    OpTensor batched_c;
    OpTensor batched_x;
    OpTensor batched_x_reverse;
    OpTensor batched_xx;
    SaberStatus single_batch(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             LstmParam<X86>& param);
};
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_LSTM_H