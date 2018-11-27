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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_ELTWISE_H

#include "saber/funcs/impl/impl_conv_eltwise.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/funcs_utils.h"
#include <cudnn.h>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderConvEltwise<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvEltwiseParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    VenderConvEltwise() {}

    ~VenderConvEltwise() {}

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     param                [conv parameters]
     */
    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvEltwiseParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvEltwiseParam<NV>& param, Context<NV>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvEltwiseParam<NV>& param);

    SaberStatus trans_weights(Tensor<NV> &target_weights,
            int stride_h, int stride_w, int group) {

        return SaberUnImplError;
    }
private:
    VenderConv2D<NV, OpDtype> _vender_conv;

};

}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
