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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_NORMALIZE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_NORMALIZE_H

#include "saber/funcs/impl/impl_normalize.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberNormalize<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        NormalizeParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberNormalize()
    {}

    ~SaberNormalize() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            NormalizeParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            NormalizeParam<ARM>& param, Context<ARM> &ctx) {
        //outputs[0]->share_from(*inputs[0]);
	    int in_c = inputs[0]->channel();
	    int in_n = inputs[0]->num();
	    Shape sh({1, 1, 1, in_c * in_n});
	    // this->_mean.re_alloc(sh);
	    // this->_variance.re_alloc(sh);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          NormalizeParam<ARM>& param);
private:
    Tensor<ARM> _mean;
    Tensor<ARM> _variance;

};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Normalize_H
