#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
from op import OpsParam, OpsRegister
from op_io import *

############################# IO define ##############################
# graph may has mult-inputs, so graph will have multi-input
OpsRegister.Register("Input").set_attr(input_shape=list(),
                                       max_len = int(),
                                       max_batch = int(),
                                       alias="NULL",
                                       data_type="NULL")

# graph out , only hold place for edge
OpsRegister.Register("Output").set_attr()

OpsRegister.Register("Split").set_attr(split_num=int())

############################# Basic Op define ##############################
# two input 
OpsRegister.Register("Dot").set_attr(axes=list())
# one or two input
# enum type {
#		 Add,
#		 Subtract,
#		 Multiply,
#		 Avg,
#		 Max
#	  }
#  note : coeff only used by caffe for "Add"
OpsRegister.Register("Eltwise").set_attr(type="Add", 
                                         coeff=list())
# list input
OpsRegister.Register("Concat").set_attr(axis=int())
# one input
OpsRegister.Register("Exp").set_attr(base=float(), 
                                     scale=float(), 
                                     shift=float())
# one input
# y = log(shift + scale * x)
OpsRegister.Register("Log").set_attr(base=float(), 
                                     scale=float(), 
                                     shift=float())
# one input
# y =  (shift + scale * x) ^ power
OpsRegister.Register("Power").set_attr(shift=float(), 
                                       scale=float(), 
                                       power=float())

# one input
OpsRegister.Register("Softmax").set_attr(axis=int())

# applies an activation parameter function to an output
# enum type:  
#		  enum type {
#			  TanH, 
#			  Sigmoid, 
# 		  }
OpsRegister.Register("Activation").set_attr(type="",
                                            clip_relu_num=int())
# Leaky version of a Rectified Linear Unit ( alpha != 0 ).
# 	f(x) = alpha * x  	 : x < 0
# 	f(x) = 		   x  	 : x >= 0
# Standard ReLU ( alpha = 0 )
#   f(x) = 0 * x     : x < 0
#   f(x) =     x     : x >= 0
#   note:  alpha is fixed value
OpsRegister.Register("ReLU").set_attr(alpha=float())
# Parametric Rectified Linear Unit
#   f(x) = alpha * x 	 : x < 0
#   f(x) = x 			 : x >= 0
#   note: alpha is learned array with the same shape as x.
#   ref: Parametric ReLU described in K. He et al, Delving Deep into Rectifiers: 
#        	<<Surpassing Human-Level Performance on ImageNet Classification>>, 2015.
OpsRegister.Register("PReLU").set_attr(channel_shared=bool())
# Exponential Linear Unit.
# 	f(x) =  alpha * (exp(x) - 1.0) 	: x < 0
#   f(x) = x 						: x >= 0
OpsRegister.Register("ELU").set_attr(alpha=int())

# dense op parameter
OpsRegister.Register("Dense").set_attr(out_dim=int(), 
                                       axis=int(), 
                                       bias_term=bool())

# dropout parameter
OpsRegister.Register("Dropout").set_attr(ratio=float()) 

OpsRegister.Register("Flatten").set_attr(start_axis=int(), 
                                         end_axis=int())

# caffe unique layer
OpsRegister.Register("Reshape").set_attr(dims=list(), 
                                         axis=int(), 
                                         num_axes=int(),
                                         layout='')

# Permutes the dimensions of the input according to a given pattern(list type)
OpsRegister.Register("Permute").set_attr(dims=list())

# Cropping op for cropping data of (1/2/3D) by using axis info
# cropping is the same as tf cropping parameter, which saved as tuple or int.
OpsRegister.Register("Cropping").set_attr(cropping=list(), 
                                          axis=int())

# slices an input layer to multiple output layers along a given dimension with given slice indices
OpsRegister.Register("Slice").set_attr(axis=int(), 
                                       slice_point=list(), 
                                       slice_dim=int(),
                                       num=int(),
                                       sections=list())


############################# Normalization Op define ##############################
# Batch normalization op
# explanation: 
#	Normalize the activations of the previous layer at each batch, 
#	i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
OpsRegister.Register("BatchNorm").set_attr(momentum=float(), 
                                           epsilon=float())

# caffe need may use scale layer after batchnorm layer which tf/mxnet/keras needn't
OpsRegister.Register("Scale").set_attr(axis=int(), 
                                       num_axes=int(), 
                                       bias_term=bool())

# Local Response Normalization op same as caffe, 
# which performs a kind of "lateral inhibition" by normalizing over local input regions
# enum NormRegion {
#	ACROSS_CHANNELS
#	WITHIN_CHANNEL
# }
OpsRegister.Register("LRN").set_attr(local_size=int(), 
                                     alpha=float(), 
                                     beta=float(), 
                                     norm_region="ACROSS_CHANNELS", 
                                     k=float())

# Mean-Variance Normalization
OpsRegister.Register("MVN").set_attr(normalize_variance=bool(), 
                                     across_channels=bool(), 
                                     epsilon=float())


############################# Pooling (1D/2D/3D) Op define ##############################
# enum type: 
#      enum method {
#           MAX, 		// [default]
#			AVG,
#           AVGEXC, average_exclude_padding_value
#			STOCHASTIC,
#      }
OpsRegister.Register("Pooling").set_attr(pool_size=list(), 
                                         strides=list(), 
                                         padding=list(), 
                                         method="MAX", 
                                         global_pooling=bool(), 
                                         cmp_out_shape_floor_as_conv=False)

# Spatial Pyramid Pooling 
# enum type: 
#      enum method {
#           MAX, 		// [default]
#			AVG,
#			STOCHASTIC,
#      }
OpsRegister.Register("SPP").set_attr(pyramid_height=int(), 
                                     method="MAX",)

############################# Convolution (1D/2D/3D) Op define ##############################
# convolution parameter
OpsRegister.Register("Convolution").set_attr(filter_num=int(), 
                                             kernel_size=list(), 
                                             strides=list(), 
                                             padding=list(), 
                                             dilation_rate=list(), 
                                             group=int(), 
                                             axis=int(), 
                                             bias_term=bool())

# Depthwise separable convolution, commonly called "separable convolution" in tf
OpsRegister.Register("DeSepConvolution").set_attr(filter_num=int(), 
                                                  kernel_size=list(), 
                                                  strides=list(), 
                                                  padding=list(), 
                                                  dilation_rate=list(), 
                                                  group=int(), 
                                                  axis=int(), 
                                                  depth_multiplier=int())

# also called transposed convolution
OpsRegister.Register("Deconvolution").set_attr(filter_num=int(), 
                                               kernel_size=list(), 
                                               strides=list(), 
                                               padding=list(), 
                                               dilation_rate=list(), 
                                               group=int(), 
                                               axis=int(), 
                                               bias_term=bool())
# DeformableConvolution
OpsRegister.Register("DeformConvolution").set_attr(filter_num=int(), 
                                                   kernel_size=list(), 
                                                   strides=list(), 
                                                   padding=list(), 
                                                   dilation_rate=list(), 
                                                   group=int(), 
                                                   axis=int(), 
                                                   bias_term=bool())


############################# Rnn Op define ##############################
# Standard  RNN (LSTM/GRU)
# enum rnn type: 
# 		 enum type {
# 			 TANH,		// base
#			 SIGMOID,	// base
# 			 RELU,		// base
#		     LSTM,
#			 GRU,
#		 }
OpsRegister.Register("RNN").set_attr(hidden_size=int(), 
                                     input_size=int(), 
                                     bias_term=bool(), 
                                     dropout=float(), 
                                     type="GRU")


############################# embedding Op define ##############################
# embedding layer, input_dim in tf or caffe means the voc num and output_dim means the emb size
OpsRegister.Register("Embedding").set_attr(input_dim=int(), 
                                           output_dim=int(), 
                                           bias_term=bool())

############################# Accuracy Op define ##############################
# NULL 


########### Object track and detection (for adu(caffe layer type)) Op define #############

# RPNProposalSSD for SSD and RPN
OpsRegister.Register("RPNProposalSSD").set_attr(**RPNProposalSSD_param())

OpsRegister.Register("RCNNDetOutputWithAttr").set_attr(**detection_output_ssd_param())

OpsRegister.Register("DFMBPSROIAlign").set_attr(**dfmb_psroi_pooling_param())

OpsRegister.Register("RCNNProposal").set_attr(**RPNProposalSSD_param())

OpsRegister.Register("ProposalImgScaleToCamCoords").set_attr(**proposal_img_scale_to_cam_coords_param())


########### VIS Op define #############

OpsRegister.Register("Axpy").set_attr()

OpsRegister.Register("PriorBox").set_attr(min_size=list(), 
                                          max_size=list(), 
                                          aspect_ratio=list(),
                                          fixed_size=list(), 
                                          fixed_ratio=list(), 
                                          density=list(),  
                                          is_flip=bool(), 
                                          is_clip=bool(), 
                                          variance=list(), 
                                          img_h=int(), 
                                          img_w=int(), 
                                          step_h=float(), 
                                          step_w=float(), 
                                          offset=float(),
                                          order=list())

# enum code_type {
#	 CORNER,
#	 CENTER_SIZE,
#	 CORNER_SIZE,
# }

OpsRegister.Register("DetectionOutput").set_attr(share_location=bool(), 
                                                 variance_encode_in_target=bool(), 
                                                 class_num=int(), 
                                                 background_id=int(), 
                                                 keep_top_k=int(), 
                                                 code_type="CORNER", 
                                                 conf_thresh=float(), 
                                                 nms_top_k=int(), 
                                                 nms_thresh=float(), 
                                                 nms_eta=float())


########### ADU Op define #############


OpsRegister.Register("Argmax").set_attr(out_max_val=bool(), 
                                        top_k=int(), 
                                        axis=int(),
                                        axis_term=bool())


########### OCR Op define #############

OpsRegister.Register("Im2Sequence").set_attr(paddings=list(),
                                             strides=list(),
                                             window_size=list(),
                                             dilations=list())


OpsRegister.Register("Cast").set_attr(in_type=int(),
                                      out_type=int())


OpsRegister.Register("Gru").set_attr(is_reverse=bool(),
                                     gate_activation="sigmoid",
                                     activation="relu",
                                     gru_formula="")

OpsRegister.Register("CtcAlign").set_attr(merge_repeated=bool(),
                                          blank=int())


########### RNN Op define #############


OpsRegister.Register("Embedding").set_attr(word_num=int(),
                                           emb_dim=int(),
                       padding_idx=int())


OpsRegister.Register("SequencePool").set_attr(pooltype="LAST")


OpsRegister.Register("SequenceConv").set_attr(filter_num=int(),
                                              kernel_size=list(), 
                                              padding_trainable=bool(),
                                              context_stride=int(),
                                              context_start=int(),
                                              context_length=int())

OpsRegister.Register("CrfDecoding").set_attr()


OpsRegister.Register("LSTM").set_attr(candidate_activation="tanh",
                                      cell_activation="tanh",
                                      gate_activation="sigmoid",
                                      is_reverse=bool(),
                                      use_peepholes=bool(),
                                      num_direction=int(),
                                      dropout_param=float(),
                                      num_layers=int(),
                                      input_activation="null")


OpsRegister.Register("MatMul").set_attr(transpose_x=bool(),
                                        transpose_y=bool(),
                                        coeff=float())


OpsRegister.Register("LayerNorm").set_attr(is_across_spatial=bool(),
                                           is_shared_channel=bool(),
                                           begin_norm_axis=int(),
                                           eps=float())

OpsRegister.Register("Resize").set_attr(method="BILINEAR_ALIGN",
                                        height_scale=float(),
                                        width_scale=float())

OpsRegister.Register("Normalize").set_attr(begin_norm_axis=int(),
                                           is_across_spatial=bool(),
                                           is_shared_channel=bool(),
                                           eps=float(),
                                           p=int())

OpsRegister.Register("Pad").set_attr(pad_c=list(),
                                     pad_h=list(),
                                     pad_w=list())


OpsRegister.Register("ShuffleChannel").set_attr(group=int())

OpsRegister.Register("RoisAnchorFeature").set_attr(min_anchor_size=float(),
                                                   num_anchor_scales=int(),
                                                   anchor_scale_pow_base=float(),
                                                   anchor_wph_ratios=list(),
                                                   num_top_iou_anchor=int(),
                                                   min_num_top_iou_anchor=int(),
                                                   iou_thr=float(),
                                                   ft_ratio_h=bool(),
                                                   ft_ratio_w=bool(),
                                                   ft_log_ratio_h=bool(),
                                                   ft_log_ratio_w=bool(),
                                                   bbox_size_add_one=bool())

OpsRegister.Register("Interp").set_attr(height=int(),
                                        width=int(),
                                        zoom_factor=int(),
                                        shrink_factor=int(),
                                        pad_beg=int(),
                                        pad_end=int())


##################################### reverse_sequence op define ############################    #########
####### it is named BatchReverseSequenceLayer in lego
#
OpsRegister.Register("ReverseSequence").set_attr()  ##no prams , no weights.

##################################### reverse op define #####################################
####### it is named BatchReverseInputLayer in lego
OpsRegister.Register("Reverse").set_attr()   ## no prams, no weights.

##################################### embedding_lg op define ################################    #####
####### it is named BatchEmbeddingLayer in lego
OpsRegister.Register("EmbeddingLg").set_attr() ## ???? is it same to Embedding?

##################################### grnn(single-layer, single-direction GRU) op define ####    #################################
####### it is named BatchGrnnLayer in lego
OpsRegister.Register("GRNN").set_attr() ## ???? is it same to RNN?

##################################### match_matrix op define ################################    #####
####### it is named BatchMatchMatrixTensorLayer in lego
OpsRegister.Register("MatchMatrix").set_attr(dim_in = int(),
                                             dim_t = int(),
                                             linear_term = bool(),
                                             bias_term = bool(),
                                             diag_init = int(),
                                             diag_init_dim_num = int(),
                                             init_low = int(),
                                             init_up = int())


##################################### var_size_conv op define ###############################    ######
####### it is named BatchVarSizeConvLayer in lego
OpsRegister.Register("VarSizeConv").set_attr()  ## it is same to convolution????
##################################### topk_pooling op define ################################    #####
###### it is named BatchTopKPoolingLayer in lego
OpsRegister.Register("TopKPooling").set_attr(top_k = int(),
                                             feat_map_num = int())

##################################### topk_avg_pooling op define ############################    #########
###### it is named BatchTopKAvgPoolingByRowLayer in lego
OpsRegister.Register("TopKAvgPooling").set_attr(top_ks = list(),
                                                feat_map_num = int(),
                                                is_pooling_by_row = bool())

##################################### extract_last op define ################################    #####
###### it is named BatchExtractLastLayer in lego,
OpsRegister.Register("SequencePool").set_attr(pooltype = str())  #no paras, no weights.


#####################################Unpadding_padding op define ############################    #########
###### it is named UnpaddingPaddingLayer in lego,
OpsRegister.Register("ConvUnpaddingPadding").set_attr()  #no paras, no weights.
# Fast-RCNN
OpsRegister.Register("AffineChannel").set_attr()  #no paras, no weights.

OpsRegister.Register("AnchorGenerator").set_attr(anchor_sizes=list(),
                                                 aspect_ratios=list(),
                                                 variances=list(),
                                                 stride=list(),
                                                 offset=float())

OpsRegister.Register("GenerateProposals").set_attr(pre_nms_top_n=int(),
                                                 post_nms_top_n=int(),
                                                 nms_thresh=float(),
                                                 min_size=float(),
                                                 eta=float())

OpsRegister.Register("RoiAlign").set_attr(spatial_scale=float(),
                                          pooled_height=int(),
                                          pooled_width=int(),
                                          sampling_ratio=int())