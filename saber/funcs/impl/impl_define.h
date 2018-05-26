/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_DEFINE_H
#define ANAKIN_SABER_FUNCS_IMPL_DEFINE_H

#include "anakin_config.h"
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/saber_funcs_param.h"
#include "saber/saber_types.h"
namespace anakin{

namespace saber{

#define DEFINE_OP_CLASS(class_name, param_name) \
template <typename TargetType, \
    DataType OpDtype = AK_FLOAT, \
    DataType inDtype = AK_FLOAT, \
    DataType outDtype = AK_FLOAT, \
    typename LayOutType_op = NCHW, \
    typename LayOutType_in = NCHW, \
    typename LayOutType_out = NCHW> \
class Saber##class_name : public ImplBase< \
    Tensor<TargetType, inDtype, LayOutType_in>, \
    Tensor<TargetType, outDtype, LayOutType_out>, \
    Tensor<TargetType, OpDtype, LayOutType_op>, \
    param_name <Tensor<TargetType, OpDtype, LayOutType_op> > > {}; \
\
template <typename TargetType, \
    DataType OpDtype = AK_FLOAT, \
    DataType inDtype = AK_FLOAT, \
    DataType outDtype = AK_FLOAT, \
    typename LayOutType_op = NCHW, \
    typename LayOutType_in = NCHW, \
    typename LayOutType_out = NCHW> \
class Vender##class_name : public ImplBase< \
    Tensor<TargetType, inDtype, LayOutType_in>, \
    Tensor<TargetType, outDtype, LayOutType_out>, \
    Tensor<TargetType, OpDtype, LayOutType_op>, \
    param_name <Tensor<TargetType, OpDtype, LayOutType_op> > > {}; 

DEFINE_OP_CLASS(Activation, ActivationParam);
DEFINE_OP_CLASS(Argmax, ArgmaxParam);
DEFINE_OP_CLASS(Axpy, AxpyParam);

DEFINE_OP_CLASS(BoxCoder, BoxCoderParam);

DEFINE_OP_CLASS(Cast, CastParam);
DEFINE_OP_CLASS(Concat, ConcatParam);
DEFINE_OP_CLASS(Conv2D, ConvParam);
DEFINE_OP_CLASS(Conv2DAct, ConvActiveParam);
DEFINE_OP_CLASS(Conv2DEltWise, ConvActiveParam);
DEFINE_OP_CLASS(Conv2DActPooling, ConvActivePoolingParam);
DEFINE_OP_CLASS(Crop, CropParam);
DEFINE_OP_CLASS(CtcAlign, CtcAlignParam);

DEFINE_OP_CLASS(Deconv2D, ConvParam);
DEFINE_OP_CLASS(Deconv2DAct, ConvActiveParam);
DEFINE_OP_CLASS(DeformableConv2D, DeformableConvParam);
DEFINE_OP_CLASS(DetectionOutput, DetectionOutputParam);

DEFINE_OP_CLASS(Eltwise, EltwiseParam);
DEFINE_OP_CLASS(EltwiseActive, EltwiseActiveParam);

DEFINE_OP_CLASS(Fc, FcParam);
DEFINE_OP_CLASS(Flatten, FlattenParam);

DEFINE_OP_CLASS(Gru, GruParam);

DEFINE_OP_CLASS(Im2Sequence, Im2SequenceParam);

DEFINE_OP_CLASS(Lrn, LrnParam);

DEFINE_OP_CLASS(MultiClassNMS, MultiClassNMSParam);
DEFINE_OP_CLASS(Mvn, MvnParam);

DEFINE_OP_CLASS(Normalize, NormalizeParam);

DEFINE_OP_CLASS(Pad, PadParam);
DEFINE_OP_CLASS(Permute, PermuteParam);
DEFINE_OP_CLASS(PermutePower, PermutePowerParam);
DEFINE_OP_CLASS(Pooling, PoolingParam);
DEFINE_OP_CLASS(PoolingWithIndex, PoolingParam);
DEFINE_OP_CLASS(Power, PowerParam);
DEFINE_OP_CLASS(Prelu, PreluParam);
DEFINE_OP_CLASS(PriorBox, PriorBoxParam);

DEFINE_OP_CLASS(Reshape, ReshapeParam);
DEFINE_OP_CLASS(Resize, ResizeParam);
DEFINE_OP_CLASS(RoiPool, RoiPoolParam);

DEFINE_OP_CLASS(Slice, SliceParam);
DEFINE_OP_CLASS(Softmax, SoftmaxParam);
DEFINE_OP_CLASS(Spp, SPPParam);

DEFINE_OP_CLASS(Transpose, TransposeParam);

DEFINE_OP_CLASS(Unpool, PoolingParam);

}
}

#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/nv_impl.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/x86_impl.h"
#endif

#ifdef USE_AMD
#endif

#ifdef USE_ARM
#endif

#endif //ANAKIN_SABER_FUNCS_IMPL_DEFINE_H
