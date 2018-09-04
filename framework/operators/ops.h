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
//#include "framework/operators/axpy.h"
#include "framework/operators/batch_norm.h"
#include "framework/operators/concat.h"
#include "framework/operators/conv_3x3.h"
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

#include "framework/operators/fusion_ops/conv_3x3_batchnorm_scale_relu.h"
#include "framework/operators/fusion_ops/conv_3x3_batchnorm_scale_relu_pool.h"
#include "framework/operators/fusion_ops/conv_3x3_relu.h"
#include "framework/operators/fusion_ops/conv_3x3_relu_pool.h"
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
#if 0
namespace anakin{
namespace graph{
    REGISTER_GRAPH_FUSION_PATTERN(DeconvRelu)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Deconvolution")
            .AddOpNode("relu_0", "ReLU")
            .AddConnect("conv_0", "relu_0")
            .CreatePattern([](VGraph* graph) {});

    REGISTER_GRAPH_FUSION_PATTERN(ConvRelu)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Convolution")
            .AddOpNode("relu_0", "ReLU")
            .AddConnect("conv_0", "relu_0")
            .CreatePattern([](VGraph* graph) {});

    REGISTER_GRAPH_FUSION_PATTERN(PermutePower)
            .Type(IN_ORDER)
            .AddOpNode("permute_0",  "Permute")
            .AddOpNode("power_0", "Power")
            .AddConnect("permute_0", "power_0")
            .CreatePattern([](VGraph* graph) {});
/*
    REGISTER_GRAPH_FUSION_PATTERN(ConvReluPool)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Convolution")
            .AddOpNode("relu_0", "ReLU")
            .AddOpNode("pooling_0", "Pooling")
            .AddConnect("conv_0", "relu_0")
            .AddConnect("relu_0", "pooling_0")
            .CreatePattern([](VGraph* graph) {});
*/
/*
    REGISTER_GRAPH_FUSION_PATTERN(ConvBatchnormScaleReluPool)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Convolution")
            .AddOpNode("batchnorm_0", "BatchNorm")
            .AddOpNode("scale_0", "Scale")
            .AddOpNode("relu_0", "ReLU")
            .AddOpNode("pooling_0", "Pooling")
            .AddConnect("conv_0", "batchnorm_0")
            .AddConnect("batchnorm_0", "scale_0")
            .AddConnect("scale_0", "relu_0")
            .AddConnect("relu_0", "pooling_0")
            .CreatePattern([](VGraph* graph) {});
*/
    REGISTER_GRAPH_FUSION_PATTERN(ConvBatchnormScaleRelu)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Convolution")
            .AddOpNode("batchnorm_0", "BatchNorm")
            .AddOpNode("scale_0", "Scale")
            .AddOpNode("relu_0", "ReLU")
            .AddConnect("conv_0", "batchnorm_0")
            .AddConnect("batchnorm_0", "scale_0")
            .AddConnect("scale_0", "relu_0")
            .CreatePattern([](VGraph* graph) {});

    REGISTER_GRAPH_FUSION_PATTERN(ConvBatchnormScale)
            .Type(IN_ORDER)
            .AddOpNode("conv_0",  "Convolution")
            .AddOpNode("batchnorm_0", "BatchNorm")
            .AddOpNode("scale_0", "Scale")
            .AddConnect("conv_0", "batchnorm_0")
            .AddConnect("batchnorm_0", "scale_0")
            .CreatePattern([](VGraph* graph) {});

    REGISTER_GRAPH_FUSION_PATTERN(EltwiseRelu)
            .Type(IN_ORDER)
            .AddOpNode("eltwise_0", "Eltwise")
            .AddOpNode("relu_0", "ReLU")
            .AddConnect("eltwise_0", "relu_0")
            .CreatePattern([](VGraph* graph) {});
}
}
#endif //0
#endif
