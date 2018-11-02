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
#include "framework/graph/llvm/fusion/graph_pattern.h"

namespace anakin {

namespace graph {

/// in straight order

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

REGISTER_GRAPH_FUSION_PATTERN(ConvReluPool)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Convolution")
.AddOpNode("relu_0", "ReLU")
.AddOpNode("pooling_0", "Pooling")
.AddConnect("conv_0", "relu_0")
.AddConnect("relu_0", "pooling_0")
.CreatePattern([](VGraph* graph) {});

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

REGISTER_GRAPH_FUSION_PATTERN(ConvBatchnorm)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Convolution")
.AddOpNode("batchnorm_0", "BatchNorm")
.AddConnect("conv_0", "batchnorm_0")
.CreatePattern([](VGraph* graph) {});

REGISTER_GRAPH_FUSION_PATTERN(EltwiseRelu)
.Type(IN_ORDER)
.AddOpNode("eltwise_0", "Eltwise")
.AddOpNode("relu_0", "ReLU")
.AddConnect("eltwise_0", "relu_0")
.CreatePattern([](VGraph* graph) {});

REGISTER_GRAPH_FUSION_PATTERN(EltwiseActivation)
.Type(IN_ORDER)
.AddOpNode("eltwise_0", "Eltwise")
.AddOpNode("prelu_0", "Activation")
.AddConnect("eltwise_0", "prelu_0")
.CreatePattern([](VGraph* graph) {});

} /* namespace graph */

} /* namespace anakin */
