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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_POOLING_H

#include "saber/funcs/impl/impl_conv_pooling.h"
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberConv2DPooling<X86, OpDtype> : public ImplBase<
        X86, OpDtype, ConvPoolingParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    typedef ImplBase<X86, OpDtype, ConvPoolingParam<X86> > Impl_conv_pool_t;

    SaberConv2DPooling():conv_pool_impl_(nullptr) {}

    ~SaberConv2DPooling() {
        if (conv_pool_impl_ != nullptr) {
            delete conv_pool_impl_;
            conv_pool_impl_ = nullptr;
        }
    }

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     param                [conv parameters]
     */
    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            ConvPoolingParam<X86>& param);

    SaberStatus trans_weights(Tensor<X86> &target_weights, Tensor<X86> &target_bias,
                              int pad_h, int pad_w, int dilation_h, int dilation_w,
                              int stride_h, int stride_w, int group) {
        return SaberUnImplError;
    }
private:
    SaberPooling<X86, OpDtype> _vender_pool;
    SaberConv2D<X86, OpDtype> _vender_conv;
    Shape _inner_shape;
    Tensor<X86> _inner_tensor;
    std::vector<Tensor<X86> *> _inner_tensor_v;
    Impl_conv_pool_t* conv_pool_impl_;
};

}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
