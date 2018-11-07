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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_POOLING_H

#include "saber/funcs/impl/impl_conv_pooling.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/impl/cuda/vender_pooling.h"
#include "saber/funcs/funcs_utils.h"
#include <cudnn.h>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderConv2DPooling<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvPoolingParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    VenderConv2DPooling() {}

    ~VenderConv2DPooling() {}

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
            ConvPoolingParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ConvPoolingParam<NV>& param, Context<NV>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
            std::vector<Tensor<NV>*>& outputs,
            ConvPoolingParam<NV>& param);
    
    SaberStatus trans_weights(Tensor<NV> &target_weights, Tensor<NV> &target_bias,
                              int pad_h, int pad_w, int dilation_h, int dilation_w,
                              int stride_h, int stride_w, int group) {
        return SaberUnImplError;
    }
private:
    VenderPooling<NV, OpDtype> _vender_pool;
    VenderConv2D<NV, OpDtype> _vender_conv;
    Shape _inner_shape;
    Tensor<NV> _inner_tensor;
    std::vector<Tensor<NV> *> _inner_tensor_v;
};

}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
