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

#ifndef ANAKIN_OPERATORS_H
#define ANAKIN_OPERATORS_H
#if 1
//#include "framework/graph/llvm/fusion/graph_pattern.h"
#include "framework/operators/activation.h"
#include "framework/operators/arg_max.h"
#include "framework/operators/axpy.h"
#include "framework/operators/batch_norm.h"
#include "framework/operators/concat.h"
#include "framework/operators/convolution.h"
#include "framework/operators/crf_decoding.h"
#include "framework/operators/crop.h"
#include "framework/operators/ctc_align.h"
#include "framework/operators/deconvolution.h"
#include "framework/operators/deformconvolution.h"
#include "framework/operators/dense.h"
#include "framework/operators/depwise_sep_convolution.h"
#include "framework/operators/detection_output.h"
#include "framework/operators/dot.h"
#include "framework/operators/dropout.h"
#include "framework/operators/eltwise_op.h"
#include "framework/operators/elu.h"
#include "framework/operators/embedding.h"
#include "framework/operators/exp.h"
#include "framework/operators/flatten.h"
#include "framework/operators/gru.h"
#include "framework/operators/im2sequence.h"
#include "framework/operators/input.h"
#include "framework/operators/log.h"
#include "framework/operators/lrn.h"
#include "framework/operators/lstm.h"
#include "framework/operators/mvn.h"
#include "framework/operators/normalize.h"
#include "framework/operators/output.h"
#include "framework/operators/permute.h"
#include "framework/operators/pooling.h"
#include "framework/operators/power.h"
#include "framework/operators/prelu.h"
#include "framework/operators/priorbox.h"
#include "framework/operators/relu.h"
#include "framework/operators/reshape.h"
#include "framework/operators/scale.h"
#include "framework/operators/sequence_pool.h"
#include "framework/operators/slice.h"
#include "framework/operators/softmax.h"
#include "framework/operators/spatial_pyramid_pooling.h"
#include "framework/operators/split.h"
#include "framework/operators/standard_rnn.h"

#include "framework/operators/fusion_ops/conv_batchnorm_scale.h"
#include "framework/operators/fusion_ops/conv_batchnorm_scale_relu.h"
#include "framework/operators/fusion_ops/conv_batchnorm_scale_relu_pool.h"
#include "framework/operators/fusion_ops/conv_relu.h"
#include "framework/operators/fusion_ops/conv_relu_pool.h"
#include "framework/operators/fusion_ops/deconv_relu.h"
#include "framework/operators/fusion_ops/eltwise_relu.h"
#include "framework/operators/fusion_ops/permute_power.h"

#endif //0

namespace anakin {
namespace ops {
} /* namespace ops */
} /* namespace anakin */

#endif
