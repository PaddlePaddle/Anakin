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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PERMUTE_H

#include "saber/funcs/impl/impl_permute.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberPermute<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        PermuteParam<Tensor<ARM, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberPermute() {
        _transpose = false;
        _need_permute = false;
    }
    ~SaberPermute() {}
    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        PermuteParam<OpTensor> &param, Context<ARM> &ctx) override {
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        PermuteParam<OpTensor> &param, Context<ARM> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, PermuteParam<OpTensor> &param) override;


public:
    int _num_axes;
    int _count;
    bool _need_permute{false};
    bool _transpose{false};
    int _trans_num;
    int _trans_w;
    int _trans_h;
    std::vector<int> _order_dims;
    std::vector<int> _new_steps;
    std::vector<int> _old_steps;

};

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PERMUTE_H
