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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H

#include "saber/funcs/impl/impl_activation.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberActivation<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        ActivationParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberActivation()
    {}

    ~SaberActivation() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ActivationParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ActivationParam<ARM>& param, Context<ARM> &ctx) {
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          ActivationParam<ARM>& param);


};

//template class SaberActivation<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H
