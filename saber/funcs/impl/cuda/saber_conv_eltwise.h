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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_ELTWISE_H

#include <vector>
#include "saber/funcs/impl/impl_conv_eltwise.h"
#include "sass_funcs.h"
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberConvEltwise<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvEltwiseParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef ImplBase<NV, OpDtype, ConvParam<NV> > Impl_conv_t;
    typedef ImplBase<NV, OpDtype, EltwiseParam<NV> > Impl_eltwise_t;

    SaberConvEltwise() {}

    ~SaberConvEltwise() {}

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

    SaberStatus trans_weights(Tensor<NV> &target_weights, Tensor<NV> &target_bias,
                              int pad_h, int pad_w, int dilation_h, int dilation_w,
                              int stride_h, int stride_w, int group);

private:
    bool _extern_trans{false};
    bool _use_k1s1p0{false};
    bool _use_k3{false};
    bool _use_vender{false};
    SaberEltwise<NV, OpDtype> _eltwise;
    SaberConv2D<NV, OpDtype> _conv;
    Shape _inner_shape;
    Tensor<NV> _inner_tensor;
    std::vector<Tensor<NV> *> _inner_tensor_v;
    int _kernel_height{0};
    int _kernel_width{0};
    std::function<void(const float*,
                       float*,
                       const float*,
                       const float*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       float,
                       float,
                       EltwiseType elt_type,
                       cudaStream_t)> dispatch_func_elt;
};
}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
