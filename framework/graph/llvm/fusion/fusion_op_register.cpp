#include "framework/graph/llvm/fusion/graph_pattern.h"
#include "anakin_config.h"
namespace anakin {

namespace graph {

/// in straight order

REGISTER_GRAPH_FUSION_PATTERN(DeconvRelu)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Deconvolution")
.AddOpNode("relu_0", "ReLU")
.AddConnect("conv_0", "relu_0")
.CreatePattern([](VGraph* graph) {});

//*
REGISTER_GRAPH_FUSION_PATTERN(DeconvBatchnormScaleRelu)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Deconvolution")
.AddOpNode("batchnorm_0", "BatchNorm")
.AddOpNode("scale_0", "Scale")
.AddOpNode("relu_0", "ReLU")
.AddConnect("conv_0", "batchnorm_0")
.AddConnect("batchnorm_0", "scale_0")
.AddConnect("scale_0", "relu_0")
.CreatePattern([](VGraph* graph) {});

REGISTER_GRAPH_FUSION_PATTERN(DeconvBatchnormScale)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Deconvolution")
.AddOpNode("batchnorm_0", "BatchNorm")
.AddOpNode("scale_0", "Scale")
.AddConnect("conv_0", "batchnorm_0")
.AddConnect("batchnorm_0", "scale_0")
.CreatePattern([](VGraph* graph) {});

//*
REGISTER_GRAPH_FUSION_PATTERN(DeconvBatchnorm)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Deconvolution")
.AddOpNode("batchnorm_0", "BatchNorm")
.AddConnect("conv_0", "batchnorm_0")
.CreatePattern([](VGraph* graph) {});
//*/

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

//*
REGISTER_GRAPH_FUSION_PATTERN(ConvBatchnorm)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Convolution")
.AddOpNode("batchnorm_0", "BatchNorm")
.AddConnect("conv_0", "batchnorm_0")
.CreatePattern([](VGraph* graph) {});
//*/
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

REGISTER_GRAPH_FUSION_PATTERN(ConvAffineChannel)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Convolution")
.AddOpNode("affine_channel_0", "AffineChannel")
.AddConnect("conv_0", "affine_channel_0")
.CreatePattern([](VGraph* graph) {});

REGISTER_GRAPH_FUSION_PATTERN(ConvAffineChannelRelu)
.Type(IN_ORDER)
.AddOpNode("conv_0",  "Convolution")
.AddOpNode("affine_channel_0", "AffineChannel")
.AddOpNode("relu_0", "ReLU")
.AddConnect("conv_0", "affine_channel_0")
.AddConnect("affine_channel_0", "relu_0")
.CreatePattern([](VGraph* graph) {});

REGISTER_GRAPH_FUSION_PATTERN(SeqConcatSeqPoolSoftSign)
.Type(IN_ORDER)
.AddOpNode("seq_concat_0",  "SequenceConcat")
.AddOpNode("seq_pool_0", "SequencePool")
.AddOpNode("soft_sign_0", "SoftSign")
.AddConnect("seq_concat_0", "seq_pool_0")
.AddConnect("seq_pool_0", "soft_sign_0")
.CreatePattern([](VGraph* graph) {});

} /* namespace graph */

} /* namespace anakin */
