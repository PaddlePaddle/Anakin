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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSoftmax<NV, OpDtype>:
    public ImplBase<NV, OpDtype, SoftmaxParam<NV>> 
{
public:
    typedef TargetWrapper<NV> API;

    SaberSoftmax() = default;

    ~SaberSoftmax() {}

    /**
     * \brief initial all cudnn resources here
     * @param inputs
     * @param outputs
     * @param param
     * @param ctx
     */
    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            SoftmaxParam<NV>& param, Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            SoftmaxParam<NV>& param, Context<NV>& ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          SoftmaxParam<NV>& param);

private:

    //! get maximum size to select which softmax kernel to call
    //! _max_dimsize is compute from shared memory size
    bool _is_continue_buf{true};
    int _max_dimsize;
    int _inner_num;
    int _outer_num;
    int _axis_size;
    int _dims;
    Tensor<NV> _input_stride;
    Tensor<NV> _output_stride;
    Tensor<NV> _valid_shape;

    Tensor<NV> _max_data;
    Tensor<NV> _sum_data;
};
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H
