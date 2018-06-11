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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_FC_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_FC_H

#include "saber/funcs/impl/impl_fc.h"
#include "saber/funcs/impl/arm/impl/sgemm_arm.h"
#include "saber/funcs/impl/arm/impl/sgemv_arm.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

//! input size: 1xk
//! output size: 1xn
//! weights size: nxk
//! bias size: 1xn

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberFc<ARM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>, \
        Tensor<ARM, outDtype, LayOutType_out>, \
        Tensor<ARM, OpDtype, LayOutType_op>, \
        FcParam<Tensor<ARM, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberFc() {}
    ~SaberFc() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        FcParam<OpTensor> &param, Context<ARM> &ctx) override {
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        FcParam<OpTensor> &param, Context<ARM> &ctx) override {

        this->_ctx = ctx;
        int threads = this->_ctx.get_act_ids().size();

        _m = inputs[0]->count_valid(0, param.axis);
        _k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        _n = param.num_output;
        int weights_size = param.weights->valid_size();
        if (_n <= 0) {
            _n = weights_size / _k;
        }
        CHECK_EQ(weights_size / _n, _k) << "weights size does not meet the input size";

        int l1_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L1_cache;
        int l2_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L2_cache;
        //! if L1 cache size is not provided, set to 31K
        l1_cache = l1_cache > 0? l1_cache : 31000;
        //! if L2 cache size is not provided, set to 2M
        l2_cache = l2_cache > 0? l2_cache : 2000000;

        _gemmer.init(l1_cache, l2_cache, _m, _n, _k, false, !param.is_transpose_weights, threads);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        FcParam<OpTensor> &param) override;


private:

    Sgemm _gemmer;
    int _m;
    int _k;
    int _n;
};

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_FC_H
