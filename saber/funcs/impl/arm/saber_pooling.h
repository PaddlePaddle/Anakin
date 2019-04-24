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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_POOLING_H

#include "saber/funcs/impl/impl_pooling.h"

namespace anakin{

namespace saber{

typedef void (*pool_func)(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          float scale, PoolingParam<ARM> param);

template <DataType OpDtype>
class SaberPooling<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        PoolingParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberPooling()
    {}

    ~SaberPooling() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            PoolingParam<ARM>& param, Context<ARM>& ctx){
      return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            PoolingParam<ARM>& param, Context<ARM> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          PoolingParam<ARM>& param) override;
private:
    pool_func _impl{nullptr};
    Tensor<ARM> _tmp_in;
    Tensor<ARM> _tmp_out;

#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
    std::string _pool_type;
#endif
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Pooling_H
