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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H

#include "saber/funcs/impl/impl_eltwise.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberEltwise<NV, OpDtype>:
    public ImplBase <
    NV, OpDtype,
    EltwiseParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;


    SaberEltwise() {}

    ~SaberEltwise() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             EltwiseParam<NV>& param,
                             Context<NV>& ctx) {
        // get context
        this->_ctx = &ctx;
        CHECK_GE(outputs.size(), 1) << "outputs size has to == 1";
        CHECK_GE(inputs.size(), 2) << "input size has to >= 2";
        _with_relu = param.has_eltwise && param.activation_param.active == Active_relu;
        _other_activation = param.has_eltwise && param.activation_param.active != Active_relu
                            && param.activation_param.active != Active_unknow;

        if (_other_activation) {
            SABER_CHECK(_saber_activation.init(inputs, outputs, param.activation_param, ctx));
        }
        int input_num = inputs.size();
        Shape coeff_shape({input_num, 1, 1, 1}, Layout_NCHW);
        if (param.operation == Eltwise_sum) {
            _coeff_d.re_alloc(coeff_shape, AK_FLOAT);

            OpDataType* coeff_data = (OpDataType*)_coeff_d.mutable_data();
            cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
            cudaMemcpyAsync(coeff_data, &param.coeff[0], sizeof(OpDataType) * input_num, cudaMemcpyHostToDevice, cuda_stream);
        }
        _inputs_d.re_alloc(coeff_shape, AK_UINT64);
        

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               EltwiseParam<NV>& param,
                               Context<NV>& ctx) {
        this->_ctx = &ctx;

        if (_other_activation) {
            SABER_CHECK(_saber_activation.create(inputs, outputs, param.activation_param, ctx));
        }

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 EltwiseParam<NV>& param) override;

private:
    //    Tensor<NV> _max_idx;
    bool _with_relu;
    bool _other_activation;
    SaberActivation<NV, OpDtype> _saber_activation;
    Tensor<NV> _coeff_d;
    Tensor<NV> _inputs_d;

};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H
