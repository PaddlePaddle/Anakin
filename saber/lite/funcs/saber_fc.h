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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_FC_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_FC_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/sgemm_arm.h"

namespace anakin{

namespace saber{

namespace lite{

//! input size: 1xk
//! output size: 1xn
//! weights size: nxk
//! bias size: 1xn
template <typename Dtype>
class SaberFc {
public:
    SaberFc() {}
    ~SaberFc() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     FcParam<Tensor<Dtype>> &param) {

        Shape shape_out = inputs[0]->valid_shape();
        int m = inputs[0]->count_valid(0, param.axis);
        int k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        int n = param.num_output;
        int weights_size = param.weights->valid_size();
        if (n <= 0) {
            n = weights_size / k;
        }
        CHECK_EQ(weights_size / n, k) << "weights size does not meet the input size";

        shape_out.resize(param.axis + 1);
        shape_out[param.axis] = n;
        return outputs[0]->set_shape(shape_out);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        FcParam<Tensor<Dtype>> &param, Context &ctx) {
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        FcParam<Tensor<Dtype>> &param, Context &ctx) {

        _ctx = ctx;
        int threads = _ctx.get_act_ids().size();

        _m = inputs[0]->count_valid(0, param.axis);
        _k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        _n = param.num_output;
        int weights_size = param.weights->valid_size();
        if (_n <= 0) {
            _n = weights_size / _k;
        }
        CHECK_EQ(weights_size / _n, _k) << "weights size does not meet the input size";

        int l1_cache = _ctx.devs[_ctx.get_device_id()]._info._L1_cache;
        int l2_cache = _ctx.devs[_ctx.get_device_id()]._info._L2_cache;
        //! if L1 cache size is not provided, set to 31K
        l1_cache = l1_cache > 0? l1_cache : 31000;
        //! if L2 cache size is not provided, set to 2M
        l2_cache = l2_cache > 0? l2_cache : 2000000;

        _gemmer.init(l1_cache, l2_cache, _m, _n, _k, false, !param.is_transpose_weights, threads);
        return SaberSuccess;
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        FcParam<Tensor<Dtype>> &param);


private:
    Context _ctx;
    Sgemm _gemmer;
    int _m;
    int _k;
    int _n;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_FC_H
