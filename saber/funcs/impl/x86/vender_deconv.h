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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_DECONV_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_deconv.h"

#ifndef USE_SGX
#include "saber/funcs/impl/x86/mkldnn_helper.h"
#endif

namespace anakin {
namespace saber {

template <DataType OpDtype>
class VenderDeconv2D<X86, OpDtype> : public ImplBase <
    X86, OpDtype, ConvParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    VenderDeconv2D() {}

    ~VenderDeconv2D() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             ConvParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               ConvParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ConvParam<X86>& param);

    SaberStatus trans_weights(Tensor<X86>& target_weights,
                              Tensor<X86>& target_bias, int pad_h, int pad_w,
                              int dilation_h, int dilation_w, int stride_h,
                              int stride_w, int group) {
        return SaberUnImplError;
    }

private:
    SaberStatus init_conv_prv(const std::vector<Tensor<X86>*>& inputs,
                              std::vector<Tensor<X86>*>& outputs, ConvParam<X86>& param);

private:
    std::shared_ptr<mkldnn::engine> _engine;
    mkldnn::algorithm _alg;
    std::vector<mkldnn::primitive> _prvs;
    std::vector<mkldnn::primitive> _prvs_weights_trans;
    std::shared_ptr<mkldnn::stream> _stream;

    mkldnn_mem_ptr _conv_in_mem;
    mkldnn_mem_ptr _conv_w_mem;
    mkldnn_mem_ptr _conv_bias_mem;
    mkldnn_mem_ptr _conv_out_mem;

    mkldnn_mem_ptr _in_mem;
    mkldnn_mem_ptr _w_mem;
    mkldnn_mem_ptr _bias_mem;
    mkldnn_mem_ptr _out_mem;

    int _in_order;
    int _out_order;


};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_DECONV_H
