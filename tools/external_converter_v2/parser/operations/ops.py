#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
from op import OpParam
from op import OpsRegister
from op_io import *

############################# IO define ##############################
# graph may has mult-inputs, so graph will have multi-input
OpsRegister.Register("Input").set_attr(input_shape=list())

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
OpsRegister.Register("Activation").set_attr(type="")
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
                                         num_axes=int())

# Permutes the dimensions of the input according to a given pattern(list type)
OpsRegister.Register("Permute").set_attr(dims=list())

# Cropping op for cropping data of (1/2/3D) by using axis info
# cropping is the same as tf cropping parameter, which saved as tuple or int.
OpsRegister.Register("Cropping").set_attr(cropping=list(), 
                                          axis=int())

# slices an input layer to multiple output layers along a given dimension with given slice indices
OpsRegister.Register("Slice").set_attr(axis=int(), 
                                       slice_point=list(), 
                                       slice_dim=int())


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


OpsRegister.Register("Normalize").set_attr(is_across_spatial=bool(), 
                                           is_shared_channel=bool(), 
                                           eps=float(), 
                                           p=int())
