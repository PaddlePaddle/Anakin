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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONV_DW_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONV_DW_H

#include <vector>
#include "saber/funcs/impl/impl_conv.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberDepthWiseConv : public ImplBase<
        ARM, OpDtype, ConvParam<ARM> > {
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    typedef void (*conv_dw_impl)(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const float* weights, const float* bias, \
                          ConvParam<ARM>& param, Context<ARM>* ctx);

    typedef void (*conv_dw_int8_impl)(const int8_t* din, int32_t* dout, \
                          int num, int chout, int hout, int wout, int chin, \
                          int hin, int win, const int8_t* weights, const int32_t* bias, \
                          ConvParam<ARM>& param, Context<ARM>* ctx, DataType out_type, const float* scale);

    SaberDepthWiseConv() = default;
    ~SaberDepthWiseConv(){}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                             std::vector<Tensor<ARM> *>& outputs,
                             ConvParam<ARM>& param, Context<ARM> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                               std::vector<Tensor<ARM> *>& outputs,
                               ConvParam<ARM>& param, Context<ARM>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM>*>& inputs,
                                 std::vector<Tensor<ARM>*>& outputs,
                                 ConvParam<ARM>& param);

private:
  conv_dw_impl _impl{nullptr};
  conv_dw_int8_impl _impl_int8{nullptr};
  std::vector<float> _w_scale;
  Tensor<ARM> _tmp_out;

};
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONV_DW_H
