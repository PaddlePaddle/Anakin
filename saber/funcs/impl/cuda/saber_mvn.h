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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MVN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MVN_H

#include "saber/funcs/impl/impl_mvn.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberMvn<NV, OpDtype>: public ImplBase<NV, OpDtype, MvnParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberMvn() {}

    ~SaberMvn() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             MvnParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               MvnParam<NV> &param,
                               Context<NV> &ctx) {
        int num = inputs[0]->num() * inputs[0]->channel();
        if (param.across_channels) {
            num = inputs[0]->num();
        }
        Shape shape = inputs[0]->valid_shape();
        for (int i = 0; i < shape.size(); i++) {
            shape[i] = 1;
        }
        shape[0] = num;
        _mean.reshape(shape);
        if (param.normalize_variance) {
            _sd.reshape(shape);
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 MvnParam<NV>  &param);

private:
    Tensor<NV> _mean;
    Tensor<NV> _sd;

};

template class SaberMvn<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MVN_H
