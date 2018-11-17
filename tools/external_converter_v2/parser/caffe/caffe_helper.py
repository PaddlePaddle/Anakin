#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from ..utils import *
from ..pbs import *


def SplitBlobName(layer_name, blob_name, blob_idx, split_idx):
    """
    Used for caffe parser.
    """
    return "_".join([layer_name, blob_name, str(blob_idx), "split", str(split_idx)])


def SplitLayerName(layer_name, blob_name, blob_idx):
    """
    Used for caffe parser.
    """
    return "_".join([layer_name, blob_name, str(blob_idx), "split"])


def UpgradeV0PaddingLayers(param, param_upgraded_pad):
    param_upgraded_pad.CopyFrom(param)
    del param_upgraded_pad.layers[:]
    # Figure out which layer each bottom blob comes from.
    blob_name_to_last_top_idx = {}
    for blob_name in param.input:
        blob_name_to_last_top_idx[blob_name] = -1
    for layer in param.layers:
        layer_param = layer.layer
        # Add the layer to the new net, unless it's a padding layer.
        if layer_param.type != "padding":
            temp_layer = param_upgraded_pad.layers.add()
            temp_layer = layer
        for idx, blob_name in enumerate(layer.bottom):
            if blob_name not in blob_name_to_last_top_idx.keys():
                print "Unknown blob input " + blob_name + " to layer " + layer.name
                exit()
            top_idx = blob_name_to_last_top_idx[blob_name]
            if top_idx == -1:
                continue
            source_layer = param.layers[top_idx]
            if source_layer.layer.type == "padding":
                #  This layer has a padding layer as input -- check that it is a conv 
                #  layer or a pooling layer and takes only one input.  Also check that 
                #  the padding layer input has only one input and one output.  Other 
                #  cases have undefined behavior in Caffe.
                layer_index = len(param_upgraded_pad.layers) - 1
                param_upgraded_pad.layers[layer_index].pad = source_layer.layer.pad
                param_upgraded_pad.layers[layer_index].bottom[idx] = source_layer.bottom[0]
        for idx, top in enumerate(layer.top):
            blob_name_to_last_top_idx[top] = idx


def UpgradeV0LayerType(type):
    if type == "accuracy":
        return V1LayerParameter.ACCURACY 
    elif type == "bnll":
        return V1LayerParameter.BNLL
    elif type == "concat":
        return V1LayerParameter.CONCAT
    elif type == "conv":
        return V1LayerParameter.CONVOLUTION
    elif type == "data":
        return V1LayerParameter.DATA
    elif type == "dropout":
        return V1LayerParameter.DROPOUT
    elif type == "euclidean_loss":
        return V1LayerParameter.EUCLIDEAN_LOSS
    elif type == "flatten": 
        return V1LayerParameter.FLATTEN
    elif type == "hdf5_data":
        return V1LayerParameter.HDF5_DATA
    elif type == "hdf5_output":
        return V1LayerParameter.HDF5_OUTPUT
    elif type == "im2col":
        return V1LayerParameter.IM2COL
    elif type == "images":
        return V1LayerParameter.IMAGE_DATA
    elif type == "infogain_loss":
        return V1LayerParameter.INFOGAIN_LOSS
    elif type == "innerproduct": 
        return V1LayerParameter.INNER_PRODUCT
    elif type == "lrn": 
        return V1LayerParameter.LRN
    elif type == "multinomial_logistic_loss": 
        return V1LayerParameter.MULTINOMIAL_LOGISTIC_LOSS
    elif type == "pool":
        return V1LayerParameter.POOLING
    elif type == "relu": 
        return V1LayerParameter.RELU
    elif type == "sigmoid": 
        return V1LayerParameter.SIGMOID
    elif type == "softmax":
        return V1LayerParameter.SOFTMAX
    elif type == "softmax_loss":
        return V1LayerParameter.SOFTMAX_LOSS
    elif type == "split":
        return V1LayerParameter.SPLIT
    elif type == "tanh":
        return V1LayerParameter.TANH
    elif type == "window_data": 
        return V1LayerParameter.WINDOW_DATA 
    else: 
        print "Unknown layer name: " + str(type) 
    return V1LayerParameter.NONE


def UpgradeV0LayerParameter(v0_layer_connection, net_param):
    layer = net_param.layer.add()
    is_fully_compatible = True
    for btm in v0_layer_connection.bottom:
        layer.bottom.append(btm)
    for top in v0_layer_connection.top:
        layer.top.append(top)
    if v0_layer_connection.HasField("layer"):
        v0_layer_param = v0_layer_connection.layer
        if v0_layer_param.HasField("name"):
            layer.name = v0_layer_param.name
        if v0_layer_param.HasField("type"):
            layer.type = UpgradeV0LayerType(v0_layer_param.type)
        for blob in v0_layer_param.blobs:
            layer.blobs.extend(blob)
        if v0_layer_param.HasField("num_output"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.num_output = v0_layer_param.num_output
            elif v0_layer_param.type == "innerproduct":
                layer_param.inner_product_param.num_output = v0_layer_param.num_output
            else:
                print "Unknown parameter num_output for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("biasterm"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.bias_term = v0_layer_param.biasterm
            elif v0_layer_param.type == "innerproduct":
                layer_param.inner_product_param.bias_term = v0_layer_param.biasterm
            else:
                print "Unknown parameter biasterm for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("weight_filler"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.weight_filler = v0_layer_param.weight_filler
            elif v0_layer_param.type == "innerproduct":
                layer_param.inner_product_param.weight_filler = v0_layer_param.weight_filler
            else:
                print "Unknown parameter weight_filler for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("bias_filler"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.bias_filler = v0_layer_param.bias_filler
            elif v0_layer_param.type == "innerproduct":
                layer_param.inner_product_param.bias_filler = v0_layer_param.bias_filler
            else:
                print "Unknown parameter bias_filler for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("pad"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.pad = v0_layer_param.pad
            elif v0_layer_param.type == "pool":
                layer_param.pooling_param.pad = v0_layer_param.pad
            else:
                print "Unknown parameter pad for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("kernelsize"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.kernel_size = v0_layer_param.kernelsize
            elif v0_layer_param.type == "pool":
                layer_param.pooling_param.kernel_size = v0_layer_param.kernelsize
            else:
                print "Unknown parameter kernelsize for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("group"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.group = v0_layer_param.group
            else:
                print "Unknown parameter group for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("stride"):
            if v0_layer_param.type == "conv":
                layer_param.convolution_param.stride = v0_layer_param.stride
            elif v0_layer_param.type == "pool":
                layer_param.pooling_param.stride = v0_layer_param.stride
            else:
                print "Unknown parameter stride for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("pool"):
            if v0_layer_param.type == "pool":
                if v0_layer_param.pool == V0LayerParameter.MAX:
                    layer_param.pooling_param.pool = PoolingParameter.MAX
                elif v0_layer_param.pool == V0LayerParameter.AVE:
                    layer_param.pooling_param.pool = PoolingParameter.AVE
                elif v0_layer_param.pool == V0LayerParameter.STOCHASTIC:
                    layer_param.pooling_param.pool = PoolingParameter.STOCHASTIC
                else:
                    print "Unknown pool method " + str(v0_layer_param.pool)
                    is_fully_compatible = False
            else:
                print "Unknown parameter pool for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("dropout_ratio"):
            if v0_layer_param.type == "dropout":
                layer_param.dropout_param.dropout_ratio = v0_layer_param.dropout_ratio
            else:
                print "Unknown parameter dropout_ratio for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("local_size"):
            if v0_layer_param.type == "lrn":
                layer_param.lrn_param.local_size = v0_layer_param.local_size
            else:
                print "Unknown parameter local_size for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False 
        if v0_layer_param.HasField("alpha"):
            if v0_layer_param.type == "lrn":
                layer_param.lrn_param.alpha = v0_layer_param.alpha
            else:
                print "Unknown parameter alpha for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("beta"):
            if v0_layer_param.type == "lrn":
                layer_param.lrn_param.beta = v0_layer_param.beta
            else:
                print "Unknown parameter beta for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("k"):
            if v0_layer_param.type == "conv":
                layer_param.lrn_param.k = v0_layer_param.k
            else:
                print "Unknown parameter k for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("source"):
            if v0_layer_param.type == "data":
                layer_param.data_param.source = v0_layer_param.source
            elif v0_layer_param.type == "hdf5_data":
                layer_param.hdf5_data_param.source = v0_layer_param.source
            elif v0_layer_param.type == "images":
                layer_param.image_data_param.source = v0_layer_param.source
            elif v0_layer_param.type == "window_data":
                layer_param.window_data_param.source = v0_layer_param.source
            elif v0_layer_param.type == "infogain_loss":
                layer_param.infogain_loss_param.source = v0_layer_param.source
            else:
                print "Unknown parameter source for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("scale"): 
            layer_param.transform_param.scale = v0_layer_param.scale
        if v0_layer_param.HasField("meanfile"): 
            layer_param.transform_param.mean_file = v0_layer_param.meanfile
        if v0_layer_param.HasField("batchsize"): 
            if v0_layer_param.type == "data":
                layer_param.data_param.batch_size = v0_layer_param.batchsize
            elif v0_layer_param.type == "hdf5_data":
                layer_param.hdf5_data_param.batch_size = v0_layer_param.batchsize
            elif v0_layer_param.type == "images":
                layer_param.image_data_param.batch_size = v0_layer_param.batchsize
            elif v0_layer_param.type == "window_data":
                layer_param.window_data_param.batch_size = v0_layer_param.batchsize
            elif v0_layer_param.type == "infogain_loss":
                layer_param.infogain_loss_param.batch_size = v0_layer_param.batchsize
            else:
                print "Unknown parameter batchsize for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("cropsize"):
            layer_param.transform_param.crop_size = v0_layer_param.cropsize
        if v0_layer_param.HasField("mirror"):
            layer_param.transform_param.mirror = v0_layer_param.mirror
        if v0_layer_param.HasField("rand_skip"):
            if v0_layer_param.type == "data":
                layer_param.data_param.rand_skip = v0_layer_param.rand_skip
            elif v0_layer_param.type == "images":
                layer_param.image_data_param.rand_skip = v0_layer_param.rand_skip
            else:
                print "Unknown parameter rand_skip for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("shuffle_images"):
            if v0_layer_param.type == "images":
                layer_param.image_data_param.shuffle = v0_layer_param.shuffle_images
            else:
                print "Unknown parameter shuffle_images for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("new_height"):
            if v0_layer_param.type == "images":
                layer_param.image_data_param.new_height = v0_layer_param.new_height
            else:
                print "Unknown parameter new_height for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("new_width"):
            if v0_layer_param.type == "images":
                layer_param.image_data_param.new_width = v0_layer_param.new_width
            else:
                print "Unknown parameter new_width for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("concat_dim"):
            if v0_layer_param.type == "concat":
                layer_param.concat_param.concat_dim = v0_layer_param.concat_dim
            else:
                print "Unknown parameter concat_dim for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("det_fg_threshold"):
            if v0_layer_param.type == "window_data":
                layer_param.window_data_param.fg_threshold = v0_layer_param.det_fg_threshold
            else:
                print "Unknown parameter det_fg_threshold for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("det_bg_threshold"):
            if v0_layer_param.type == "window_data":
                layer_param.window_data_param.bg_threshold = v0_layer_param.det_bg_threshold
            else:
                print "Unknown parameter det_bg_threshold for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("det_fg_fraction"):
            if v0_layer_param.type == "window_data":
                layer_param.window_data_param.fg_fraction = v0_layer_param.det_fg_fraction
            else:
                print "Unknown parameter det_fg_fraction for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("det_context_pad"):
            if v0_layer_param.type == "window_data":
                layer_param.window_data_param.context_pad = v0_layer_param.det_context_pad
            else:
                print "Unknown parameter det_context_pad for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("det_crop_mode"):
            if v0_layer_param.type == "window_data":
                layer_param.window_data_param.crop_mode = v0_layer_param.det_crop_mode
            else:
                print "Unknown parameter crop_mode for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
        if v0_layer_param.HasField("hdf5_output_param"):
            if v0_layer_param.type == "hdf5_output":
                layer_param.hdf5_output_param = v0_layer_param.hdf5_output_param
            else:
                print "Unknown parameter hdf5_output_param for layer type: " + str(v0_layer_param.type)
                is_fully_compatible = False
    return is_fully_compatible


def UpgradeV0Net(original_param, net_param):
    # First upgrade padding layers to padded conv layers.
    v0_net_param = NetParameter()
    UpgradeV0PaddingLayers(original_param, v0_net_param)
    # Now upgrade layer parameters.
    is_fully_compatible = True
    # net_param->Clear();
    del net_param.layer[:]
    if v0_net_param.HasField("name"):
        net_param.name = v0_net_param.name
    for layer in v0_net_param.layers:
        is_fully_compatible = is_fully_compatible and UpgradeV0LayerParameter(layer, net_param)
    del net_param.input[:]
    for input in v0_net_param.input:
        net_param.input.append(input)
    del net_param.input_dim[:]
    for dim in v0_net_param.input_dim:
        net_param.input_dim.append(dim)
    if v0_net_param.HasField("force_backward"):
        net_param.force_backward = v0_net_param.force_backward
    return is_fully_compatible


def NetNeedsV0ToV1Upgrade(net_param): 
    layers = net_param.layers or net_param.layer
    for layer in layers:
        if proto_has_field(layer, "layer"):
            return True
    return False


def NetNeedsV1ToV2Upgrade(net_param):
    return len(net_param.layers) > 0


def NetNeedsDataUpgrade(net_param):
    layers = net_param.layers or net_param.layer
    for layer in layers:
        if layer.type == V1LayerParameter.DATA:
            # DataParameter
            layer_param = layer.data_param
            if proto_has_field(layer_param, "scale"):
                return True
            if proto_has_field(layer_param, "mean_file"):
                return True
            if proto_has_field(layer_param, "crop_size"):
                return True
            if proto_has_field(layer_param, "mirror"):
                return True
        if layer.type == V1LayerParameter.IMAGE_DATA:
            # ImageDataParameter 
            layer_param = layer.image_data_param
            if proto_has_field(layer_param, "scale"):
                return True
            if proto_has_field(layer_param, "mean_file"):
                return True
            if proto_has_field(layer_param, "crop_size"):
                return True
            if proto_has_field(layer_param, "mirror"):
                return True
        if layer.type == V1LayerParameter.WINDOW_DATA:
            # WindowDataParameter
            layer_param = layer.window_data_param
            if proto_has_field(layer_param, "scale"):
                return True
            if proto_has_field(layer_param, "mean_file"):
                return True
            if proto_has_field(layer_param, "crop_size"):
                return True
            if proto_has_field(layer_param, "mirror"):
                return True
    return False


def NetNeedsInputUpgrade(net_param):
    return len(net_param.input) > 0


def NetNeedsBatchNormUpgrade(net_param):
    layers = net_param.layers or net_param.layer
    for layer in layers:
        # Check if BatchNorm layers declare three parameters, as required by the previous BatchNorm layer definition.
        if layer.type == "BatchNorm" and len(layer.param) == 3:
            return True
    return False


def UpgradeNetInput(net_param):
	has_shape = len(net_param.input_shape) > 0;
	has_dim = len(net_param.input_dim) > 0
	if has_shape or has_dim:
		layer = net_param.layer.add()
		layer.name = "input"
		layer.type = "Input"
		input_param = layer.input_param
		for idx, input in enumerate(net_param.input):
			layer.top.append(input)
			if has_shape:
				input_param.shape.add().CopyFrom(net_param.input_shape[idx])
			else:
				shape = input_param.shape.add()
				first_dim = 4*idx
				last_dim = first_dim + 4
				for j in range(4):
					shape.dim.append(net_param.input_dim[first_dim + j])

		# Swap input layer to beginning of net to satisfy layer dependencies.
		layers = net_param.layers or net_param.layer
		layers_tmp = []
		layer_input_index = len(layers) - 1
		layers_tmp.append(layers[layer_input_index])
		for i in range(len(layers) - 1):
			layers_tmp.append(layers[i])
		if net_param.layers:
			del net_param.layers[:]
			net_param.layers.extend(layers_tmp)
		else:
			del net_param.layer[:]
			net_param.layer.extend(layers_tmp)
	del net_param.input[:] 
	del net_param.input_shape[:] 
	del net_param.input_dim[:]

UpgradeV1LayerTypeDict = {
        V1LayerParameter.NONE: "",
        V1LayerParameter.ABSVAL: "AbsVal",
        V1LayerParameter.ACCURACY: "Accuracy",
        V1LayerParameter.ARGMAX: "ArgMax",
        V1LayerParameter.BNLL: "BNLL",
        V1LayerParameter.CONCAT: "Concat",
        V1LayerParameter.CONTRASTIVE_LOSS: "ContrastiveLoss",
        V1LayerParameter.CONVOLUTION: "Convolution",
        V1LayerParameter.DECONVOLUTION: "Deconvolution",
        V1LayerParameter.DATA: "Data",
        V1LayerParameter.DROPOUT: "Dropout",
        V1LayerParameter.DUMMY_DATA: "DummyData",
        V1LayerParameter.EUCLIDEAN_LOSS: "EuclideanLoss",
        V1LayerParameter.ELTWISE: "Eltwise",
        V1LayerParameter.EXP: "Exp",
        V1LayerParameter.FLATTEN: "Flatten",
        V1LayerParameter.HDF5_DATA: "HDF5Data",
        V1LayerParameter.HDF5_OUTPUT: "HDF5Output",
        V1LayerParameter.HINGE_LOSS: "HingeLoss",
        V1LayerParameter.IM2COL: "Im2col",
        V1LayerParameter.IMAGE_DATA: "ImageData",
        V1LayerParameter.INFOGAIN_LOSS: "InfogainLoss",
        V1LayerParameter.INNER_PRODUCT: "InnerProduct",
        V1LayerParameter.LRN: "LRN",
        V1LayerParameter.MEMORY_DATA: "MemoryData",
        V1LayerParameter.MULTINOMIAL_LOGISTIC_LOSS: "MultinomialLogisticLoss",
        V1LayerParameter.MVN: "MVN",
        V1LayerParameter.POOLING: "Pooling",
        V1LayerParameter.POWER: "Power",
        V1LayerParameter.RELU: "ReLU",
        V1LayerParameter.SIGMOID: "Sigmoid",
        V1LayerParameter.SIGMOID_CROSS_ENTROPY_LOSS: "SigmoidCrossEntropyLoss",
        V1LayerParameter.SILENCE: "Silence",
        V1LayerParameter.SOFTMAX: "Softmax",
        V1LayerParameter.SOFTMAX_LOSS: "SoftmaxWithLoss",
        V1LayerParameter.SPLIT: "Split",
        V1LayerParameter.SLICE: "Slice",
        V1LayerParameter.TANH: "TanH",
        V1LayerParameter.WINDOW_DATA: "WindowData",
        V1LayerParameter.THRESHOLD: "Threshold",
        #V1LayerParameter.BATCH_NORM_KYLE:"BatchNorm"
}


def UpgradeV1LayerType(type):
    if type == 22231:
        return "BatchNorm"
    if type in UpgradeV1LayerTypeDict.keys():
        return UpgradeV1LayerTypeDict[type]
    else:
        print "Unknown V1LayerParameter layer type: " + str(type)
    return ""


def UpgradeV1LayerParameter(v1_layer_param, net_param):
    layer_param = net_param.layer.add()
    is_fully_compatible = True
    for btm in v1_layer_param.bottom:
        layer_param.bottom.append(btm)
    for top in v1_layer_param.top:
        layer_param.top.append(top)
    if v1_layer_param.HasField("name"):
        layer_param.name = v1_layer_param.name
    if v1_layer_param.HasField("type"):
        layer_param.type = UpgradeV1LayerType(v1_layer_param.type)
    del layer_param.blobs[:]
    for blob in v1_layer_param.blobs:
        layer_param.blobs.append(blob)
    if v1_layer_param.HasField("accuracy_param"):
        layer_param.accuracy_param.CopyFrom(v1_layer_param.accuracy_param)
    if v1_layer_param.HasField("argmax_param"):
        layer_param.argmax_param.CopyFrom(v1_layer_param.argmax_param)
    if v1_layer_param.HasField("concat_param"):
        layer_param.concat_param.CopyFrom(v1_layer_param.concat_param)
    if v1_layer_param.HasField("contrastive_loss_param"):
        layer_param.contrastive_loss_param.CopyFrom(v1_layer_param.contrastive_loss_param)
    if v1_layer_param.HasField("convolution_param"):
        layer_param.convolution_param.CopyFrom(v1_layer_param.convolution_param)
    if v1_layer_param.HasField("data_param"):
        layer_param.data_param.CopyFrom(v1_layer_param.data_param)
    if v1_layer_param.HasField("dropout_param"):
        layer_param.dropout_param.CopyFrom(v1_layer_param.dropout_param)
    if v1_layer_param.HasField("dummy_data_param"):
        layer_param.dummy_data_param.CopyFrom(v1_layer_param.dummy_data_param)
    if v1_layer_param.HasField("eltwise_param"):
        layer_param.eltwise_param.CopyFrom(v1_layer_param.eltwise_param)
    if v1_layer_param.HasField("exp_param"):
        layer_param.exp_param.CopyFrom(v1_layer_param.exp_param)
    if v1_layer_param.HasField("hdf5_data_param"):
        layer_param.hdf5_data_param.CopyFrom(v1_layer_param.hdf5_data_param)
    if v1_layer_param.HasField("hdf5_output_param"):
        layer_param.hdf5_output_param.CopyFrom(v1_layer_param.hdf5_output_param)
    if v1_layer_param.HasField("hinge_loss_param"):
        layer_param.hinge_loss_param.CopyFrom(v1_layer_param.hinge_loss_param)
    if v1_layer_param.HasField("image_data_param"):
        layer_param.image_data_param.CopyFrom(v1_layer_param.image_data_param)
    if v1_layer_param.HasField("infogain_loss_param"):
        layer_param.infogain_loss_param.CopyFrom(v1_layer_param.infogain_loss_param)
    if v1_layer_param.HasField("inner_product_param"):
        layer_param.inner_product_param.CopyFrom(v1_layer_param.inner_product_param)
    if v1_layer_param.HasField("lrn_param"):
        layer_param.lrn_param.CopyFrom(v1_layer_param.lrn_param)
    if v1_layer_param.HasField("memory_data_param"):
        layer_param.memory_data_param.CopyFrom(v1_layer_param.memory_data_param)
    if v1_layer_param.HasField("mvn_param"):
        layer_param.mvn_param.CopyFrom(v1_layer_param.mvn_param)
    if v1_layer_param.HasField("pooling_param"):
        layer_param.pooling_param.CopyFrom(v1_layer_param.pooling_param)
    if v1_layer_param.HasField("power_param"):
        layer_param.power_param.CopyFrom(v1_layer_param.power_param)
    if v1_layer_param.HasField("relu_param"):
        layer_param.relu_param.CopyFrom(v1_layer_param.relu_param)
    if v1_layer_param.HasField("sigmoid_param"):
        layer_param.sigmoid_param.CopyFrom(v1_layer_param.sigmoid_param)
    if v1_layer_param.HasField("softmax_param"):
        layer_param.softmax_param.CopyFrom(v1_layer_param.softmax_param)
    if v1_layer_param.HasField("slice_param"):
        layer_param.slice_param.CopyFrom(v1_layer_param.slice_param)
    if v1_layer_param.HasField("tanh_param"):
        layer_param.tanh_param.CopyFrom(v1_layer_param.tanh_param)
    if v1_layer_param.HasField("threshold_param"):
        layer_param.threshold_param.CopyFrom(v1_layer_param.threshold_param)
    if v1_layer_param.HasField("window_data_param"):
        layer_param.window_data_param.CopyFrom(v1_layer_param.window_data_param)
    if v1_layer_param.HasField("transform_param"):
        layer_param.transform_param.CopyFrom(v1_layer_param.transform_param)
    if v1_layer_param.HasField("loss_param"):
        layer_param.loss_param.CopyFrom(v1_layer_param.loss_param)
    if v1_layer_param.HasField("layer"):
        print "Input NetParameter has V0 layer -- ignoring."
        is_fully_compatible = false
    return is_fully_compatible


def UpgradeV1Net(v1_net_param, net_param):
    is_fully_compatible = True
    # net_param->clear_layers()
    # net_param->clear_layer()
    del net_param.layers[:]
    del net_param.layer[:]
    for layer in v1_net_param.layers:
        if not UpgradeV1LayerParameter(layer, net_param):
            print "Upgrade of input layer " + layer.name + " failed."
            is_fully_compatible = False
    return is_fully_compatible


def UpgradeNetBatchNorm(net_param):
    for layer in net_param.layer:
        if (layer.type == "BatchNorm") and (len(layer.param) == 3):
            for param in layer.param: 
                param.lr_mult = 0.0
                param.decay_mult = 0.0
