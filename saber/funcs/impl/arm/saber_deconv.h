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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDeconv2D<ARM, OpDtype> :
        public ImplBase<ARM, OpDtype, ConvParam<ARM> > {
public:
    typedef ImplBase<ARM, OpDtype, ConvParam<ARM> > Impl_t;
    SaberDeconv2D() = default;

    ~SaberDeconv2D() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                             std::vector<Tensor<ARM> *>& outputs,
                             ConvParam<ARM>& param, Context<ARM>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                               std::vector<Tensor<ARM> *>& outputs,
                               ConvParam<ARM>& param, Context<ARM> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM>*>& inputs,
                                 std::vector<Tensor<ARM>*>& outputs,
                                 ConvParam<ARM>& param);

    SaberStatus trans_weights(Tensor<ARM> &target_weights,
                              Tensor<ARM> &target_bias,
                              int in_channel, int out_channel,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group){
       return SaberUnImplError;
    }
private:
    int _m;
    int _n;
    int _k;
    int _kh;
    int _kw;
    Tensor<ARM> _tmp_in;
    Tensor<ARM> _tmp_out;
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_DECONV_H
