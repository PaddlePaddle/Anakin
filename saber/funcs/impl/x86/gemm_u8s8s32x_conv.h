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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_GEMM_U8S8S32X_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_GEMM_U8S8S32X_CONV_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_macro.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"

namespace anakin {
namespace saber {

using namespace jit;

class GemmU8S8S32XConv :
    public ImplBase<
        X86,
        AK_INT8,
        ConvEltwiseParam<X86> > {
public:
    typedef typename DataTrait<X86, AK_INT8>::Dtype OpDataType;

    GemmU8S8S32XConv()
        : weights_internal_(nullptr), acc_(nullptr), col_(nullptr),
          bias_internal_(nullptr), ws_(nullptr), ws_per_thread_(0) {
        memset(&jcp, 0, sizeof(jcp));
    }

    ~GemmU8S8S32XConv() {
        if (bias_internal_ != nullptr) {
            delete bias_internal_;
            bias_internal_ = nullptr;
        }
        if (weights_internal_ != nullptr) {
            delete weights_internal_;
            weights_internal_ = nullptr;
        }
        if (ws_ != nullptr) {
            delete ws_;
            ws_ = nullptr;
        }
        if (acc_ != nullptr) {
            delete acc_;
            acc_ = nullptr;
        }
        if (col_ != nullptr) {
            delete col_;
            col_ = nullptr;
        }
        std::vector<float>().swap(scale_);
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*> &inputs,
                             std::vector<Tensor<X86>*> &outputs,
                             ConvEltwiseParam<X86> &param,
                             Context<X86> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86>*> &inputs,
                               std::vector<Tensor<X86>*> &outputs,
                               ConvEltwiseParam<X86> &param,
                               Context<X86> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*> &inputs,
                                 std::vector<Tensor<X86>*> &outputs,
                                 ConvEltwiseParam<X86> &param);


private:
    Tensor<X86>* weights_internal_;
    Tensor<X86>* bias_internal_;
    int *ws_;
    size_t ws_per_thread_;
    int32_t *acc_;
    unsigned char *col_;
    jit_conv_conf_t jcp;

    // scale for quantization
    std::vector<float> scale_;

    virtual SaberStatus init_conf(jit_conv_conf_t &jcp,
                                  const std::vector<Tensor<X86>*> &inputs,
                                  std::vector<Tensor<X86>*> &outputs,
                                  ConvEltwiseParam<X86> &param);

    virtual SaberStatus check_conf(const jit_conv_conf_t &jcp,
                                   const std::vector<Tensor<X86>*> &inputs,
                                   std::vector<Tensor<X86>*> &outputs,
                                   ConvEltwiseParam<X86> &param);

    virtual SaberStatus im2col_u8(const jit_conv_conf_t &jcp,
                                  const unsigned char * im,
                                  unsigned char * col);

    virtual SaberStatus weight_reorder_oihw2hwio(Tensor<X86>* in,
                                                 Tensor<X86>* out);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_GEMM_U8S8S32X_CONV_H
