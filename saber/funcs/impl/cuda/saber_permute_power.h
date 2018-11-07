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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_POWER_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_POWER_H

#include "saber/funcs/impl/impl_permute_power.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPermutePower<NV, OpDtype>:\
    public ImplBase<
        NV,
        OpDtype,
        PermutePowerParam<NV> > {

public:

    SaberPermutePower() {}

    ~SaberPermutePower() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             PermutePowerParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               PermutePowerParam<NV> &param,
                               Context<NV> &ctx) {
        _num_axes = inputs[0]->shape().size();
        PermuteParam<NV> permute_param = param.permute_param;
        for (int i = 0; i < _num_axes; i++) {
            if (std::find(_order_dims.begin(), _order_dims.end(), permute_param.order[i]) == _order_dims.end()) {
                _order_dims.push_back(permute_param.order[i]);
            }
        }
        CHECK_EQ(_num_axes, _order_dims.size());

        // set _need_permute
        _need_permute = false;
        for (int i = 0; i < _num_axes; ++i) {
            if (permute_param.order[i] != i) {
                _need_permute = true;
                break;
            }
        }
        Shape order_shape({_num_axes, 1, 1, 1});
        CUDA_CHECK(cudaPeekAtLastError());
        _permute_order.re_alloc(order_shape, OpDtype);
        _old_steps.re_alloc(order_shape, OpDtype);
        _new_steps.re_alloc(order_shape, OpDtype);
        _out_valid_shape.re_alloc(order_shape, OpDtype);
        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();
        Shape out_valid_shape  = outputs[0]->valid_shape();
        cudaMemcpy((void*)(_old_steps.mutable_data()), &in_stride[0],
                   sizeof(int) * _num_axes, cudaMemcpyHostToDevice);
        cudaMemcpy(_new_steps.mutable_data(), &out_stride[0],
                   sizeof(int) * _num_axes, cudaMemcpyHostToDevice);
        cudaMemcpy(_permute_order.mutable_data(), &(permute_param.order[0]),
                   sizeof(int) * _num_axes, cudaMemcpyHostToDevice);
        cudaMemcpy(_out_valid_shape.mutable_data(), &out_valid_shape[0],
                   sizeof(int) * _num_axes, cudaMemcpyHostToDevice);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 PermutePowerParam<NV> &permute_param);

private:
    int _num_axes;
    bool _need_permute;
    std::vector<int> _order_dims;
    Tensor<NV> _permute_order;
    Tensor<NV> _out_valid_shape;
    Tensor<NV> _old_steps;
    Tensor<NV> _new_steps;
};

template class SaberPermutePower<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_POWER_H
