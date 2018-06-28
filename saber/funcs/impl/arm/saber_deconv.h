/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"
#include "saber/funcs/impl/arm/impl/sgemm_arm.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{


template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberDeconv2D<ARM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        ConvParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberDeconv2D();

    ~SaberDeconv2D();

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 ConvParam<OpTensor> &conv_param) override;

    SaberStatus set_activation(bool flag) {
        _flag_relu = flag;
        return SaberSuccess;
    }

private:
    Sgemm _gemmer;
    bool _flag_relu{false};
    bool _bias_term{true};
    int _kw;
    int _kh;
    int _m;
    int _n;
    int _k;
    size_t _workspace_fwd_sizes{0};
    std::shared_ptr<Buffer<ARM>> _workspace_data{nullptr};
};


} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_DECONV_H
