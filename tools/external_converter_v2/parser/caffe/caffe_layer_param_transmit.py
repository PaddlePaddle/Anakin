#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
try:
    from google.protobuf.internal.containers import RepeatedScalarFieldContainer as repeat_container # 3.0.0a3+
except ImportError:
    pass
try:
    from google.protobuf.pyext._message import RepeatedScalarContainer as repeat_container # 3.5.1 +
except ImportError:
    pass
from ..operations import OpsParam, OpsRegister
from ..logger import *
from ..pbs import *


def is_has_proto_key(param_pkg, key_name):
    """
    Judge if param_pkg has field key_name
    """
    for field in param_pkg.DESCRIPTOR.fields:
        if field.name == key_name:
            return True
    return False


def ParserFeedDecorator(OpName):
    """
    Decorator for parser
    """
    def warpper(Parser):
        """
        Decorator warpper function
        """
        def warpper_args(args):
            """
            Decorator warpper for args
            """
            Parser(args)
            # args[2] hold tensors
            assert len(args[2]) <= 3, " The number of tensors(real: %d) weights must <= 3 " % (len(args[2]))
            if len(args[2]) == 3:
                OpsRegister()[OpName].weight_1 = args[2][0]
                OpsRegister()[OpName].weight_2 = args[2][1]
                OpsRegister()[OpName].weight_3 = args[2][2]
            if len(args[2]) == 2:
                OpsRegister()[OpName].weight_1 = args[2][0]
                OpsRegister()[OpName].weight_2 = args[2][1]
            if len(args[2]) == 1:
                OpsRegister()[OpName].weight_1 = args[2][0]
            # args[0] hold node_io proto object
            OpsRegister()[OpName].feed_node_attr(args[0])
            # args[3] hold opIO proto object
            args[3].set_name(OpName)
            # fill node_io with opIO
            args[0].set_op(args[3]())
        return warpper_args
    return warpper

# common


def NotNeededInInference(args):
    """
    Not need to parsing
    """
    # args is tuple object
    node_io = args[0]
    layer = args[1]
    tensors = args[2]
    logger(verbose.INFO).feed("Layer type(", layer.name, " : ", layer.type, ") with ", \
            len(tensors), " tensors  not needed in inference.")


@ParserFeedDecorator("BatchNorm")
def Parser_batch_norm(args):
    layer = args[1]
    # parser caffe parameter
    batch_norm_param = layer.batch_norm_param
    OpsRegister()["BatchNorm"].momentum = batch_norm_param.moving_average_fraction
    OpsRegister()["BatchNorm"].epsilon = batch_norm_param.eps


@ParserFeedDecorator("Concat")
def Parser_concat(args):
    layer = args[1]
    # parser caffe parameter
    concat_param = layer.concat_param
    OpsRegister()["Concat"].axis = concat_param.axis

@ParserFeedDecorator("Resize")
def Parser_resize(args):
    layer = args[1]
    # parser caffe parameter
    resize_param = layer.resize_param
    if resize_param.HasField("out_width_scale"):
        OpsRegister()["Resize"].width_scale = resize_param.out_width_scale
    if resize_param.HasField("out_height_scale"):
        OpsRegister()["Resize"].height_scale = resize_param.out_height_scale
    if resize_param.HasField("out_width"):
        OpsRegister()["Resize"].out_width = resize_param.out_width
    if resize_param.HasField("out_height"):
        OpsRegister()["Resize"].out_height = resize_param.out_height
    method = ""
    if resize_param.type == ResizeParameter.BILINEAR_ALIGN:
        method = "BILINEAR_ALIGN"
    elif resize_param.type == ResizeParameter.BILINEAR_NO_ALIGN:
        method = "BILINEAR_NO_ALIGN"
    else:
        method = "RESIZE_CUSTOM"
    OpsRegister()["Resize"].method = method



@ParserFeedDecorator("DeformConvolution")
def Parser_deformable_convolution(args):
    layer = args[1]
    # parser caffe parameter
    convolution_param = layer.convolution_param
    OpsRegister()["DeformConvolution"].filter_num = convolution_param.num_output
    kernel_size = []
    if type(convolution_param.kernel_size) == repeat_container: # support for old version caffe proto
        if len(convolution_param.kernel_size):
            if len(convolution_param.kernel_size) == 1:
                kernel_size = list([convolution_param.kernel_size[0], convolution_param.kernel_size[0]])
            else:
                kernel_size = list(convolution_param.kernel_size)
        else:
            kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    elif convolution_param.kernel_size > 0:
        kernel_size = list([convolution_param.kernel_size, convolution_param.kernel_size])
    else:
        kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    OpsRegister()["DeformConvolution"].kernel_size = kernel_size
    strides = []
    if type(convolution_param.stride) == repeat_container:
        if len(convolution_param.stride):
            if len(convolution_param.stride) == 1:
                strides = list([convolution_param.stride[0], convolution_param.stride[0]])
            else:
                strides = list(convolution_param.stride)
        else:
            strides = [convolution_param.stride_h, convolution_param.stride_w]
    elif convolution_param.stride > 0:
        strides = [convolution_param.stride, convolution_param.stride]
    else:
        strides = [convolution_param.stride_h, convolution_param.stride_w]
    if strides[0] == 0:
        strides[0] = 1
        strides[1] = 1
    OpsRegister()["DeformConvolution"].strides = strides
    paddings = []
    if type(convolution_param.pad) == repeat_container:
        if len(convolution_param.pad):
            if len(convolution_param.pad) == 1:
                paddings = list([convolution_param.pad[0], convolution_param.pad[0]])
            else:
                paddings = list(convolution_param.pad)
        else:
            paddings = [convolution_param.pad_h, convolution_param.pad_w]
    elif convolution_param.pad > 0:
        paddings = list([convolution_param.pad, convolution_param.pad])
    else:
        paddings = [convolution_param.pad_h, convolution_param.pad_w]
    OpsRegister()["DeformConvolution"].padding = paddings
    if is_has_proto_key(convolution_param, "dilation"):
        if len(convolution_param.dilation) == 0:
            OpsRegister()["DeformConvolution"].dilation_rate = list([1, 1])
        elif len(convolution_param.dilation) == 1:
            OpsRegister()["DeformConvolution"].dilation_rate = list([convolution_param.dilation[0], convolution_param.dilation[0]])
        else:
            OpsRegister()["DeformConvolution"].dilation_rate = list(convolution_param.dilation)
    else:
        OpsRegister()["DeformConvolution"].dilation_rate = list([1, 1])
    OpsRegister()["DeformConvolution"].group = convolution_param.group
    if is_has_proto_key(convolution_param, "axis"):
        OpsRegister()["DeformConvolution"].axis = convolution_param.axis
    else:
        OpsRegister()["DeformConvolution"].axis = 1
    OpsRegister()["DeformConvolution"].bias_term = convolution_param.bias_term


@ParserFeedDecorator("Deconvolution")
def Parser_deconvolution(args):
    layer = args[1]
    # parser caffe parameter
    convolution_param = layer.convolution_param
    OpsRegister()["Deconvolution"].filter_num = convolution_param.num_output
    kernel_size = []
    if type(convolution_param.kernel_size) == repeat_container: # support for old version caffe proto
        if len(convolution_param.kernel_size):
            if len(convolution_param.kernel_size) == 1:
                kernel_size = list([convolution_param.kernel_size[0], convolution_param.kernel_size[0]])
            else:
                kernel_size = list(convolution_param.kernel_size)
        else:
            kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    elif convolution_param.kernel_size > 0:
        kernel_size = list([convolution_param.kernel_size, convolution_param.kernel_size])
    else:
        kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    OpsRegister()["Deconvolution"].kernel_size = kernel_size
    strides = []
    if type(convolution_param.stride) == repeat_container:
        if len(convolution_param.stride):
            if len(convolution_param.stride) == 1:
                strides = list([convolution_param.stride[0], convolution_param.stride[0]])
            else:
                strides = list(convolution_param.stride)
        else:
            strides = [convolution_param.stride_h, convolution_param.stride_w]
    elif convolution_param.stride > 0:
        strides = [convolution_param.stride, convolution_param.stride]
    else:
        strides = [convolution_param.stride_h, convolution_param.stride_w]
    if strides[0] == 0:
        strides[0] = 1
        strides[1] = 1
    OpsRegister()["Deconvolution"].strides = strides
    paddings = []
    if type(convolution_param.pad) == repeat_container:
        if len(convolution_param.pad):
            if len(convolution_param.pad) == 1:
                paddings = list([convolution_param.pad[0], convolution_param.pad[0]])
            else:
                paddings = list(convolution_param.pad)
        else:
            paddings = [convolution_param.pad_h, convolution_param.pad_w]
    elif convolution_param.pad > 0:
        paddings = list([convolution_param.pad, convolution_param.pad])
    else:
        paddings = [convolution_param.pad_h, convolution_param.pad_w]
    OpsRegister()["Deconvolution"].padding = paddings
    if is_has_proto_key(convolution_param, "dilation"):
        if len(convolution_param.dilation) == 0:
            OpsRegister()["Deconvolution"].dilation_rate = list([1, 1])
        elif len(convolution_param.dilation) == 1:
            OpsRegister()["Deconvolution"].dilation_rate = list([convolution_param.dilation[0], convolution_param.dilation[0]])
        else:
            OpsRegister()["Deconvolution"].dilation_rate = list(convolution_param.dilation)
    else:
        OpsRegister()["Deconvolution"].dilation_rate = list([1, 1])
    OpsRegister()["Deconvolution"].group = convolution_param.group
    if is_has_proto_key(convolution_param, "axis"):
        OpsRegister()["Deconvolution"].axis = convolution_param.axis
    else:
        OpsRegister()["Deconvolution"].axis = 1
    OpsRegister()["Deconvolution"].bias_term = convolution_param.bias_term


@ParserFeedDecorator("Convolution")
def Parser_convolution(args):
    layer = args[1]
    # parser caffe parameter
    convolution_param = layer.convolution_param
    OpsRegister()["Convolution"].filter_num = convolution_param.num_output
    kernel_size = []
    if type(convolution_param.kernel_size) == repeat_container: # support for old version caffe proto
        if len(convolution_param.kernel_size):
            if len(convolution_param.kernel_size) == 1:
                kernel_size = list([convolution_param.kernel_size[0], convolution_param.kernel_size[0]])
            else:
                kernel_size = list(convolution_param.kernel_size)
        else:
            kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    elif convolution_param.kernel_size > 0:
        kernel_size = list([convolution_param.kernel_size, convolution_param.kernel_size])
    else:
        kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    OpsRegister()["Convolution"].kernel_size = kernel_size
    strides = []
    if type(convolution_param.stride) == repeat_container:
        if len(convolution_param.stride):
            if len(convolution_param.stride) == 1:
                strides = list([convolution_param.stride[0], convolution_param.stride[0]])
            else:
                strides = list(convolution_param.stride)
        else:
            strides = [convolution_param.stride_h, convolution_param.stride_w]
    elif convolution_param.stride > 0:
        strides = [convolution_param.stride, convolution_param.stride]
    else:
        strides = [convolution_param.stride_h, convolution_param.stride_w]
    if strides[0] == 0:
        strides[0] = 1
        strides[1] = 1
    OpsRegister()["Convolution"].strides = strides
    paddings = []
    if type(convolution_param.pad) == repeat_container:
        if len(convolution_param.pad):
            if len(convolution_param.pad) == 1:
                paddings = list([convolution_param.pad[0], convolution_param.pad[0]])
            else:
                paddings = list(convolution_param.pad)
        else:
            paddings = [convolution_param.pad_h, convolution_param.pad_w]
    elif convolution_param.pad > 0:
        paddings = list([convolution_param.pad, convolution_param.pad])
    else:
        paddings = [convolution_param.pad_h, convolution_param.pad_w]
    OpsRegister()["Convolution"].padding = paddings
    if is_has_proto_key(convolution_param, "dilation"):
        if len(convolution_param.dilation) == 0:
            OpsRegister()["Convolution"].dilation_rate = list([1, 1])
        elif len(convolution_param.dilation) == 1:
            OpsRegister()["Convolution"].dilation_rate = list([convolution_param.dilation[0], convolution_param.dilation[0]])
        else:
            OpsRegister()["Convolution"].dilation_rate = list(convolution_param.dilation)
    else:
        OpsRegister()["Convolution"].dilation_rate = list([1, 1])
    OpsRegister()["Convolution"].group = convolution_param.group
    if is_has_proto_key(convolution_param, "axis"):
        OpsRegister()["Convolution"].axis = convolution_param.axis
    else:
        OpsRegister()["Convolution"].axis = 1
    OpsRegister()["Convolution"].bias_term = convolution_param.bias_term

@ParserFeedDecorator("Convolution")
def Parser_convolutiondepthwise(args):
    layer = args[1]
    # parser caffe parameter
    convolution_param = layer.convolution_param
    OpsRegister()["Convolution"].filter_num = convolution_param.num_output
    kernel_size = []
    if type(convolution_param.kernel_size) == repeat_container: # support for old version caffe proto
        if len(convolution_param.kernel_size):
            if len(convolution_param.kernel_size) == 1:
                kernel_size = list([convolution_param.kernel_size[0], convolution_param.kernel_size[0]])
            else:
                kernel_size = list(convolution_param.kernel_size)
        else:
            kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    elif convolution_param.kernel_size > 0:
        kernel_size = list([convolution_param.kernel_size, convolution_param.kernel_size])
    else:
        kernel_size = [convolution_param.kernel_h, convolution_param.kernel_w]
    OpsRegister()["Convolution"].kernel_size = kernel_size
    strides = []
    if type(convolution_param.stride) == repeat_container:
        if len(convolution_param.stride):
            if len(convolution_param.stride) == 1:
                strides = list([convolution_param.stride[0], convolution_param.stride[0]])
            else:
                strides = list(convolution_param.stride)
        else:
            strides = [convolution_param.stride_h, convolution_param.stride_w]
    elif convolution_param.stride > 0:
        strides = [convolution_param.stride, convolution_param.stride]
    else:
        strides = [convolution_param.stride_h, convolution_param.stride_w]
    if strides[0] == 0:
        strides[0] = 1
        strides[1] = 1
    OpsRegister()["Convolution"].strides = strides
    paddings = []
    if type(convolution_param.pad) == repeat_container:
        if len(convolution_param.pad):
            if len(convolution_param.pad) == 1:
                paddings = list([convolution_param.pad[0], convolution_param.pad[0]])
            else:
                paddings = list(convolution_param.pad)
        else:
            paddings = [convolution_param.pad_h, convolution_param.pad_w]
    elif convolution_param.pad > 0:
        paddings = list([convolution_param.pad, convolution_param.pad])
    else:
        paddings = [convolution_param.pad_h, convolution_param.pad_w]
    OpsRegister()["Convolution"].padding = paddings
    if is_has_proto_key(convolution_param, "dilation"):
        if len(convolution_param.dilation) == 0:
            OpsRegister()["Convolution"].dilation_rate = list([1, 1])
        elif len(convolution_param.dilation) == 1:
            OpsRegister()["Convolution"].dilation_rate = list([convolution_param.dilation[0], convolution_param.dilation[0]])
        else:
            OpsRegister()["Convolution"].dilation_rate = list(convolution_param.dilation)
    else:
        OpsRegister()["Convolution"].dilation_rate = list([1, 1])
    OpsRegister()["Convolution"].group = convolution_param.num_output
    if is_has_proto_key(convolution_param, "axis"):
        OpsRegister()["Convolution"].axis = convolution_param.axis
    else:
        OpsRegister()["Convolution"].axis = 1
    OpsRegister()["Convolution"].bias_term = convolution_param.bias_term

@ParserFeedDecorator("Cropping")
def Parser_crop(args):
    layer = args[1]
    # parser caffe parameter
    crop_param = layer.crop_param
    OpsRegister()["Cropping"].cropping = list(crop_param.offset)
    OpsRegister()["Cropping"].axis = crop_param.axis


@ParserFeedDecorator("Dropout")
def Parser_dropout(args):
    layer = args[1]
    # parser caffe parameter
    dropout_param = layer.dropout_param
    OpsRegister()["Dropout"].ratio = dropout_param.dropout_ratio


@ParserFeedDecorator("Eltwise")
def Parser_eltwise(args):
    layer = args[1]
    # parser caffe parameter
    eltwise_param = layer.eltwise_param
    Eltwise_type = ""
    if eltwise_param.operation == EltwiseParameter.SUM:
        Eltwise_type = "Add"
    elif eltwise_param.operation == EltwiseParameter.MAX:
        Eltwise_type = "Max"
    else:
        Eltwise_type = "Multiply"
    OpsRegister()["Eltwise"].type = Eltwise_type
    if len(eltwise_param.coeff):
        OpsRegister()["Eltwise"].coeff = list(eltwise_param.coeff)


@ParserFeedDecorator("ELU")
def Parser_elu(args):
    layer = args[1]
    # parser caffe parameter
    elu_param = layer.elu_param
    OpsRegister()["ELU"].alpha = elu_param.alpha


@ParserFeedDecorator("Embedding")
def Parser_embed(args):
    layer = args[1]
    # parser caffe parameter
    embed_param = layer.embed_param
    OpsRegister()["Embedding"].output_dim = embed_param.num_output
    OpsRegister()["Embedding"].input_dim = embed_param.input_dim
    OpsRegister()["Embedding"].bias_term = embed_param.bias_term


@ParserFeedDecorator("Exp")
def Parser_exp(args):
    layer = args[1]
    # parser caffe parameter
    exp_param = layer.exp_param
    OpsRegister()["Exp"].base = exp_param.base
    OpsRegister()["Exp"].scale = exp_param.scale
    OpsRegister()["Exp"].shift = exp_param.shift


@ParserFeedDecorator("Flatten")
def Parser_flatten(args):
    layer = args[1]
    # parser caffe parameter
    flatten_param = layer.flatten_param
    OpsRegister()["Flatten"].start_axis = flatten_param.axis
    OpsRegister()["Flatten"].end_axis = flatten_param.end_axis


@ParserFeedDecorator("Dense")
def Parser_innerproduct(args):
    layer = args[1]
    # parser caffe parameter
    tensors = args[2]
    weight = tensors[0]
    inner_product_param = layer.inner_product_param
    OpsRegister()["Dense"].axis = inner_product_param.axis # weight().shape.dim.value[2]
    OpsRegister()["Dense"].out_dim = inner_product_param.num_output # weight().shape.dim.value[3]
    OpsRegister()["Dense"].bias_term = inner_product_param.bias_term


@ParserFeedDecorator("Log")
def Parser_log(args):
    layer = args[1]
    # parser caffe parameter
    log_param = layer.log_param
    OpsRegister()["Log"].base = log_param.base
    OpsRegister()["Log"].scale = log_param.scale
    OpsRegister()["Log"].shift = log_param.shift


@ParserFeedDecorator("LRN")
def Parser_lrn(args):
    layer = args[1]
    # parser caffe parameter
    lrn_param = layer.lrn_param
    OpsRegister()["LRN"].local_size = lrn_param.local_size
    OpsRegister()["LRN"].alpha = lrn_param.alpha
    OpsRegister()["LRN"].beta = lrn_param.beta
    norm_region = ""
    if lrn_param.norm_region == LRNParameter.ACROSS_CHANNELS:
        norm_region = "ACROSS_CHANNELS"
    else:
        norm_region = "WITHIN_CHANNEL"
    OpsRegister()["LRN"].norm_region = norm_region
    OpsRegister()["LRN"].k = lrn_param.k


@ParserFeedDecorator("MVN")
def Parser_mvn(args):
    layer = args[1]
    # parser caffe parameter
    mvn_param = layer.mvn_param
    OpsRegister()["MVN"].normalize_variance = mvn_param.normalize_variance
    OpsRegister()["MVN"].across_channels = mvn_param.across_channels
    OpsRegister()["MVN"].epsilon = mvn_param.eps


@ParserFeedDecorator("Pooling")
def Parser_pooling(args):
    layer = args[1]
    # parser caffe parameter
    pooling_param = layer.pooling_param
    pool_size = []
    if pooling_param.HasField('kernel_size'):
        pool_size = [pooling_param.kernel_size, pooling_param.kernel_size]
    else:
        pool_size = [pooling_param.kernel_h, pooling_param.kernel_w]
    OpsRegister()["Pooling"].pool_size = pool_size
    strides = []
    if pooling_param.HasField("stride"):
        strides = [pooling_param.stride, pooling_param.stride]
    else:
        strides = [pooling_param.stride_h, pooling_param.stride_w]
    OpsRegister()["Pooling"].strides = strides
    padding = []
    if pooling_param.HasField('pad'):
        padding = [pooling_param.pad, pooling_param.pad]
    else:
        padding = [pooling_param.pad_h, pooling_param.pad_w]
    OpsRegister()["Pooling"].padding = padding
    method = ""
    if pooling_param.pool == PoolingParameter.MAX:
        method = "MAX"
    elif pooling_param.pool == PoolingParameter.AVE:
        method = "AVG"
    else:
        method = "STOCHASTIC"
    OpsRegister()["Pooling"].method = method
    OpsRegister()["Pooling"].global_pooling = pooling_param.global_pooling
    floor_mode = False
    if is_has_proto_key(pooling_param, "cmp_out_shape_floor_as_conv"):
        floor_mode = pooling_param.cmp_out_shape_floor_as_conv
    if is_has_proto_key(pooling_param, "ceil_mode"):
        floor_mode = floor_mode or (not pooling_param.ceil_mode)
    OpsRegister()["Pooling"].cmp_out_shape_floor_as_conv = floor_mode


@ParserFeedDecorator("Power")
def Parser_power(args):
    layer = args[1]
    # parser caffe parameter
    power_param = layer.power_param
    OpsRegister()["Power"].shift = power_param.shift
    OpsRegister()["Power"].scale = power_param.scale
    OpsRegister()["Power"].power = power_param.power


@ParserFeedDecorator("Activation")
def Parser_prelu(args):
    layer = args[1]
    # parser caffe parameter
    prelu_param = layer.prelu_param
    OpsRegister()["Activation"].type = "PReLU"
    OpsRegister()["Activation"].channel_shared = prelu_param.channel_shared


@ParserFeedDecorator("RNN")
def Parser_rnn_ori(args):
    layer = args[1]
    # parser caffe parameter
    tensors = args[2]
    weight = tensors[0]

    recurrent_param = layer.recurrent_param
    OpsRegister()["RNN"].hidden_size = recurrent_param.num_output
    OpsRegister()["RNN"].input_size = weight().shape.dim.value[2] # don't have related value, decide by weights
    OpsRegister()["RNN"].bias_term = True if len(tensors) == 2 else False
    OpsRegister()["RNN"].dropout = 0.0
    OpsRegister()["RNN"].type = "TANH" # default ??? tanh ???


@ParserFeedDecorator("RNN")
def Parser_rnn_lstm(args):
    layer = args[1]
    # parser caffe parameter
    tensors = args[2]
    weight = tensors[0]
    recurrent_param = layer.recurrent_param
    OpsRegister()["RNN"].hidden_size = recurrent_param.num_output
    OpsRegister()["RNN"].input_size = weight().shape.dim.value[2] # don't have related value, decide by weights
    OpsRegister()["RNN"].bias_term = True if len(tensors) == 2 else False
    OpsRegister()["RNN"].dropout = 0.0
    OpsRegister()["RNN"].type = "LSTM"


@ParserFeedDecorator("ReLU")
def Parser_relu(args):
    layer = args[1]
    # parser caffe parameter
    relu_param = layer.relu_param
    OpsRegister()["ReLU"].alpha = relu_param.negative_slope


@ParserFeedDecorator("SPP")
def Parser_spp(args):
    layer = args[1]
    # parser caffe parameter
    spp_param = layer.spp_param
    OpsRegister()["SPP"].pyramid_height = spp_param.pyramid_height
    method = ""
    if spp_param.pool == SPPParameter.MAX:
        method = "MAX"
    elif spp_param.pool == SPPParameter.AVE:
        method = "AVG"
    else:
        method = "STOCHASTIC"
    OpsRegister()["SPP"].method = method


@ParserFeedDecorator("Slice")
def Parser_slice(args):
    layer = args[1]
    # parser caffe parameter
    slice_param = layer.slice_param
    OpsRegister()["Slice"].axis = slice_param.axis
    OpsRegister()["Slice"].slice_point = list(slice_param.slice_point)
    OpsRegister()["Slice"].slice_dim = slice_param.slice_dim


@ParserFeedDecorator("Activation")
def Parser_tanh(args):
    # parser caffe parameter
    logger(verbose.INFO).feed("Layer  in tanh")
    OpsRegister()["Activation"].type = "TanH"


@ParserFeedDecorator("Activation")
def Parser_sigmoid(args):
    # parser caffe parameter
    logger(verbose.INFO).feed("Layer  in Sigmoid")
    OpsRegister()["Activation"].type = "Sigmoid"


@ParserFeedDecorator("Softmax")
def Parser_softmax(args):
    layer = args[1]
    # parser caffe parameter
    softmax_param = layer.softmax_param
    OpsRegister()["Softmax"].axis = softmax_param.axis


@ParserFeedDecorator("Input")
def Parser_input(args):
    layer = args[1]
    # parser caffe parameter
    input_param = layer.input_param
    OpsRegister()["Input"].input_shape = list(input_param.shape[0].dim)
    #OpsRegister()["Input"].input_num = len(input_param.shape)
    #for shape in input_param.shape:
    #   OpsRegister()["Input"].input_shape.append(list(shape.dim))


@ParserFeedDecorator("Permute")
def Parser_permute(args):
    layer = args[1]
    # parser caffe parameter
    permute_param = layer.permute_param
    OpsRegister()["Permute"].dims = list(permute_param.order)


@ParserFeedDecorator("Scale")
def Parser_scale(args):
    layer = args[1]
    # parser caffe parameter
    scale_param = layer.scale_param
    OpsRegister()["Scale"].axis = scale_param.axis
    OpsRegister()["Scale"].num_axes = scale_param.num_axes
    OpsRegister()["Scale"].bias_term = scale_param.bias_term


@ParserFeedDecorator("Reshape")
def Parser_reshape(args):
    layer = args[1]
    # parser caffe parameter
    reshape_param = layer.reshape_param
    shape = list(reshape_param.shape.dim)
    OpsRegister()["Reshape"].dims = shape
    OpsRegister()["Reshape"].axis = reshape_param.axis
    OpsRegister()["Reshape"].num_axes = reshape_param.num_axes
    if len(shape) == 4:
        layout = 'NCHW'
    elif len(shape) == 3:
        layout = 'NHW'
    OpsRegister()["Reshape"].layout = layout

@ParserFeedDecorator("Split")
def Parser_split(args):
    layer = args[1]
    # parser caffe parameter
    top = layer.top
    OpsRegister()["Split"].split_num = len(top)

@ParserFeedDecorator("ShuffleChannel")
def Parser_ShuffleChannel(args):
    layer = args[1]
    # parser caffe parameter
    shufflechannel_param = layer.shuffle_channel_param
    OpsRegister()["ShuffleChannel"].group = shufflechannel_param.group

@ParserFeedDecorator("Coord2Patch")
def Parser_Coord2Patch(args):
    layer = args[1]
    # parser caffe parameter
    coord2patch_param = layer.coord2patch_param
    OpsRegister()["Coord2Patch"].img_h = coord2patch_param.img_h
    OpsRegister()["Coord2Patch"].output_h = coord2patch_param.output_h
    OpsRegister()["Coord2Patch"].output_w = coord2patch_param.output_w

@ParserFeedDecorator("RPNProposalSSD")
def Parser_rpn_proposal_ssd(args):
    layer = args[1]
    # parser caffe parameter
    detect_output_ssd = layer.detection_output_ssd_param
    OpsRegister()["RPNProposalSSD"].threshold = list(detect_output_ssd.threshold)
    OpsRegister()["RPNProposalSSD"].channel_per_scale = detect_output_ssd.channel_per_scale
    OpsRegister()["RPNProposalSSD"].class_name_list = list(detect_output_ssd.class_name_list)
    OpsRegister()["RPNProposalSSD"].num_class = detect_output_ssd.num_class
    OpsRegister()["RPNProposalSSD"].refine_out_of_map_bbox = detect_output_ssd.refine_out_of_map_bbox
    OpsRegister()["RPNProposalSSD"].class_indexes = list(detect_output_ssd.class_indexes)
    OpsRegister()["RPNProposalSSD"].heat_map_a = list(detect_output_ssd.heat_map_a)
    OpsRegister()["RPNProposalSSD"].heat_map_b = list(detect_output_ssd.heat_map_b)
    OpsRegister()["RPNProposalSSD"].threshold_objectness = detect_output_ssd.threshold_objectness
    OpsRegister()["RPNProposalSSD"].proposal_min_sqrt_area = list(detect_output_ssd.proposal_min_sqrt_area)
    OpsRegister()["RPNProposalSSD"].proposal_max_sqrt_area = list(detect_output_ssd.proposal_max_sqrt_area)
    OpsRegister()["RPNProposalSSD"].bg_as_one_of_softmax = detect_output_ssd.bg_as_one_of_softmax
    OpsRegister()["RPNProposalSSD"].use_target_type_rcnn = detect_output_ssd.use_target_type_rcnn
    OpsRegister()["RPNProposalSSD"].im_width = detect_output_ssd.im_width
    OpsRegister()["RPNProposalSSD"].im_height = detect_output_ssd.im_height
    OpsRegister()["RPNProposalSSD"].rpn_proposal_output_score = detect_output_ssd.rpn_proposal_output_score
    OpsRegister()["RPNProposalSSD"].regress_agnostic = detect_output_ssd.regress_agnostic
    OpsRegister()["RPNProposalSSD"].allow_border = detect_output_ssd.allow_border
    OpsRegister()["RPNProposalSSD"].allow_border_ratio = detect_output_ssd.allow_border_ratio
    OpsRegister()["RPNProposalSSD"].bbox_size_add_one = detect_output_ssd.bbox_size_add_one
    OpsRegister()["RPNProposalSSD"].read_width_scale = detect_output_ssd.read_width_scale
    OpsRegister()["RPNProposalSSD"].read_height_scale = detect_output_ssd.read_height_scale
    OpsRegister()["RPNProposalSSD"].read_height_offset = detect_output_ssd.read_height_offset
    OpsRegister()["RPNProposalSSD"].min_size_h = detect_output_ssd.min_size_h
    OpsRegister()["RPNProposalSSD"].min_size_w = detect_output_ssd.min_size_w
    if detect_output_ssd.min_size_mode == DetectionOutputSSDParameter.HEIGHT_AND_WIDTH:
        OpsRegister()["RPNProposalSSD"].min_size_mode = "HEIGHT_AND_WIDTH"
    else:
        OpsRegister()["RPNProposalSSD"].min_size_mode = "HEIGHT_OR_WIDTH"
    # parsing nms_param pkg
    nms_param = detect_output_ssd.nms_param
    OpsRegister()["RPNProposalSSD"].need_nms = nms_param.need_nms
    OpsRegister()["RPNProposalSSD"].overlap_ratio = list(nms_param.overlap_ratio)
    OpsRegister()["RPNProposalSSD"].top_n = list(nms_param.top_n)
    OpsRegister()["RPNProposalSSD"].add_score = nms_param.add_score
    OpsRegister()["RPNProposalSSD"].max_candidate_n = list(nms_param.max_candidate_n)
    OpsRegister()["RPNProposalSSD"].use_soft_nms = list(nms_param.use_soft_nms)
    OpsRegister()["RPNProposalSSD"].nms_among_classes = nms_param.nms_among_classes
    OpsRegister()["RPNProposalSSD"].voting = list(nms_param.voting)
    OpsRegister()["RPNProposalSSD"].vote_iou = list(nms_param.vote_iou)
    OpsRegister()["RPNProposalSSD"].nms_gpu_max_n_per_time = nms_param.nms_gpu_max_n_per_time
    # parsing gen_anchor_param pkg
    gen_anchor_param = detect_output_ssd.gen_anchor_param
    OpsRegister()["RPNProposalSSD"].base_size = gen_anchor_param.base_size
    OpsRegister()["RPNProposalSSD"].ratios = list(gen_anchor_param.ratios)
    OpsRegister()["RPNProposalSSD"].scales = list(gen_anchor_param.scales)
    OpsRegister()["RPNProposalSSD"].anchor_width = list(gen_anchor_param.anchor_width)
    OpsRegister()["RPNProposalSSD"].anchor_height = list(gen_anchor_param.anchor_height)
    OpsRegister()["RPNProposalSSD"].anchor_x1 = list(gen_anchor_param.anchor_x1)
    OpsRegister()["RPNProposalSSD"].anchor_y1 = list(gen_anchor_param.anchor_y1)
    OpsRegister()["RPNProposalSSD"].anchor_x2 = list(gen_anchor_param.anchor_x2)
    OpsRegister()["RPNProposalSSD"].anchor_y2 = list(gen_anchor_param.anchor_y2)
    OpsRegister()["RPNProposalSSD"].zero_anchor_center = gen_anchor_param.zero_anchor_center
    # parsing kpts_param pkg
    kpts_param = detect_output_ssd.kpts_param
    OpsRegister()["RPNProposalSSD"].kpts_exist_bottom_idx = kpts_param.kpts_exist_bottom_idx
    OpsRegister()["RPNProposalSSD"].kpts_reg_bottom_idx = kpts_param.kpts_reg_bottom_idx
    OpsRegister()["RPNProposalSSD"].kpts_reg_as_classify = kpts_param.kpts_reg_as_classify
    OpsRegister()["RPNProposalSSD"].kpts_classify_width = kpts_param.kpts_classify_width
    OpsRegister()["RPNProposalSSD"].kpts_classify_height = kpts_param.kpts_classify_height
    OpsRegister()["RPNProposalSSD"].kpts_reg_norm_idx_st = kpts_param.kpts_reg_norm_idx_st
    OpsRegister()["RPNProposalSSD"].kpts_st_for_each_class = list(kpts_param.kpts_st_for_each_class)
    OpsRegister()["RPNProposalSSD"].kpts_ed_for_each_class = list(kpts_param.kpts_ed_for_each_class)
    OpsRegister()["RPNProposalSSD"].kpts_classify_pad_ratio = kpts_param.kpts_classify_pad_ratio
    # parsing atrs_param pkg
    atrs_param = detect_output_ssd.atrs_param
    OpsRegister()["RPNProposalSSD"].atrs_reg_bottom_idx = atrs_param.atrs_reg_bottom_idx
    OpsRegister()["RPNProposalSSD"].atrs_reg_norm_idx_st = atrs_param.atrs_reg_norm_idx_st
    atrs_norm_type_vec = []
    for norm_type in list(atrs_param.atrs_norm_type):
        if norm_type == ATRSParameter.WIDTH:
            atrs_norm_type_vec.append("WIDTH")
        elif norm_type == ATRSParameter.HEIGHT:
            atrs_norm_type_vec.append("HEIGHT")
        elif norm_type == ATRSParameter.WIDTH_LOG:
            atrs_norm_type_vec.append("WIDTH_LOG")
        elif norm_type == ATRSParameter.HEIGHT_LOG:
            atrs_norm_type_vec.append("HEIGHT_LOG")
        else:
            atrs_norm_type_vec.append("NONE")
    OpsRegister()["RPNProposalSSD"].atrs_norm_type = atrs_norm_type_vec
    # parsing ftrs_param pkg
    ftrs_param = detect_output_ssd.ftrs_param
    OpsRegister()["RPNProposalSSD"].ftrs_bottom_idx = ftrs_param.ftrs_bottom_idx
    # parsing spmp_param pkg
    spmp_param = detect_output_ssd.spmp_param
    OpsRegister()["RPNProposalSSD"].spmp_bottom_idx = spmp_param.spmp_bottom_idx
    OpsRegister()["RPNProposalSSD"].spmp_class_aware = list(spmp_param.spmp_class_aware)
    OpsRegister()["RPNProposalSSD"].spmp_label_width = list(spmp_param.spmp_label_width)
    OpsRegister()["RPNProposalSSD"].spmp_label_height = list(spmp_param.spmp_label_height)
    OpsRegister()["RPNProposalSSD"].spmp_pad_ratio = list(spmp_param.spmp_pad_ratio)
    # parsing cam3d_param pkg
    cam3d_param = detect_output_ssd.cam3d_param
    OpsRegister()["RPNProposalSSD"].cam3d_bottom_idx = cam3d_param.cam3d_bottom_idx

    bbox_reg = layer.bbox_reg_param
    OpsRegister()["RPNProposalSSD"].bbox_mean = list(bbox_reg.bbox_mean)
    OpsRegister()["RPNProposalSSD"].bbox_std = list(bbox_reg.bbox_std)


@ParserFeedDecorator("RCNNDetOutputWithAttr")
def Parser_rcnn_net_output_with_attr(args):
    layer = args[1]
    # parser caffe parameter
    detect_output_ssd = layer.detection_output_ssd_param
    OpsRegister()["RCNNDetOutputWithAttr"].threshold = list(detect_output_ssd.threshold)
    OpsRegister()["RCNNDetOutputWithAttr"].channel_per_scale = detect_output_ssd.channel_per_scale
    OpsRegister()["RCNNDetOutputWithAttr"].class_name_list = str(detect_output_ssd.class_name_list)
    OpsRegister()["RCNNDetOutputWithAttr"].num_class = detect_output_ssd.num_class
    OpsRegister()["RCNNDetOutputWithAttr"].refine_out_of_map_bbox = detect_output_ssd.refine_out_of_map_bbox
    OpsRegister()["RCNNDetOutputWithAttr"].class_indexes = list(detect_output_ssd.class_indexes)
    OpsRegister()["RCNNDetOutputWithAttr"].heat_map_a = list(detect_output_ssd.heat_map_a)
    OpsRegister()["RCNNDetOutputWithAttr"].heat_map_b = list(detect_output_ssd.heat_map_b)
    OpsRegister()["RCNNDetOutputWithAttr"].threshold_objectness = detect_output_ssd.threshold_objectness
    OpsRegister()["RCNNDetOutputWithAttr"].proposal_min_sqrt_area = list(detect_output_ssd.proposal_min_sqrt_area)
    OpsRegister()["RCNNDetOutputWithAttr"].proposal_max_sqrt_area = list(detect_output_ssd.proposal_max_sqrt_area)
    OpsRegister()["RCNNDetOutputWithAttr"].bg_as_one_of_softmax = detect_output_ssd.bg_as_one_of_softmax
    OpsRegister()["RCNNDetOutputWithAttr"].use_target_type_rcnn = detect_output_ssd.use_target_type_rcnn
    OpsRegister()["RCNNDetOutputWithAttr"].im_width = detect_output_ssd.im_width
    OpsRegister()["RCNNDetOutputWithAttr"].im_height = detect_output_ssd.im_height
    OpsRegister()["RCNNDetOutputWithAttr"].rpn_proposal_output_score = detect_output_ssd.rpn_proposal_output_score
    OpsRegister()["RCNNDetOutputWithAttr"].regress_agnostic = detect_output_ssd.regress_agnostic
    OpsRegister()["RCNNDetOutputWithAttr"].allow_border = detect_output_ssd.allow_border
    OpsRegister()["RCNNDetOutputWithAttr"].allow_border_ratio = detect_output_ssd.allow_border_ratio
    OpsRegister()["RCNNDetOutputWithAttr"].bbox_size_add_one = detect_output_ssd.bbox_size_add_one
    OpsRegister()["RCNNDetOutputWithAttr"].read_width_scale = detect_output_ssd.read_width_scale
    OpsRegister()["RCNNDetOutputWithAttr"].read_height_scale = detect_output_ssd.read_height_scale
    OpsRegister()["RCNNDetOutputWithAttr"].read_height_offset = detect_output_ssd.read_height_offset
    OpsRegister()["RCNNDetOutputWithAttr"].min_size_h = detect_output_ssd.min_size_h
    OpsRegister()["RCNNDetOutputWithAttr"].min_size_w = detect_output_ssd.min_size_w
    if detect_output_ssd.min_size_mode == DetectionOutputSSDParameter.HEIGHT_AND_WIDTH:
        OpsRegister()["RCNNDetOutputWithAttr"].min_size_mode = "HEIGHT_AND_WIDTH"
    else:
        OpsRegister()["RCNNDetOutputWithAttr"].min_size_mode = "HEIGHT_OR_WIDTH"
    # parsing nms_param pkg
    nms_param = detect_output_ssd.nms_param
    OpsRegister()["RCNNDetOutputWithAttr"].need_nms = nms_param.need_nms
    OpsRegister()["RCNNDetOutputWithAttr"].overlap_ratio = list(nms_param.overlap_ratio)
    OpsRegister()["RCNNDetOutputWithAttr"].top_n = list(nms_param.top_n)
    OpsRegister()["RCNNDetOutputWithAttr"].add_score = nms_param.add_score
    OpsRegister()["RCNNDetOutputWithAttr"].max_candidate_n = list(nms_param.max_candidate_n)
    OpsRegister()["RCNNDetOutputWithAttr"].use_soft_nms = list(nms_param.use_soft_nms)
    OpsRegister()["RCNNDetOutputWithAttr"].nms_among_classes = nms_param.nms_among_classes
    OpsRegister()["RCNNDetOutputWithAttr"].voting = list(nms_param.voting)
    OpsRegister()["RCNNDetOutputWithAttr"].vote_iou = list(nms_param.vote_iou)
    OpsRegister()["RCNNDetOutputWithAttr"].nms_gpu_max_n_per_time = nms_param.nms_gpu_max_n_per_time
    # parsing gen_anchor_param pkg
    gen_anchor_param = detect_output_ssd.gen_anchor_param
    OpsRegister()["RCNNDetOutputWithAttr"].base_size = gen_anchor_param.base_size
    OpsRegister()["RCNNDetOutputWithAttr"].ratios = list(gen_anchor_param.ratios)
    OpsRegister()["RCNNDetOutputWithAttr"].scales = list(gen_anchor_param.scales)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_width = list(gen_anchor_param.anchor_width)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_height = list(gen_anchor_param.anchor_height)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_x1 = list(gen_anchor_param.anchor_x1)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_y1 = list(gen_anchor_param.anchor_y1)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_x2 = list(gen_anchor_param.anchor_x2)
    OpsRegister()["RCNNDetOutputWithAttr"].anchor_y2 = list(gen_anchor_param.anchor_y2)
    OpsRegister()["RCNNDetOutputWithAttr"].zero_anchor_center = gen_anchor_param.zero_anchor_center
    # parsing kpts_param pkg
    kpts_param = detect_output_ssd.kpts_param
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_exist_bottom_idx = kpts_param.kpts_exist_bottom_idx
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_reg_bottom_idx = kpts_param.kpts_reg_bottom_idx
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_reg_as_classify = kpts_param.kpts_reg_as_classify
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_classify_width = kpts_param.kpts_classify_width
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_classify_height = kpts_param.kpts_classify_height
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_reg_norm_idx_st = kpts_param.kpts_reg_norm_idx_st
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_st_for_each_class = list(kpts_param.kpts_st_for_each_class)
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_ed_for_each_class = list(kpts_param.kpts_ed_for_each_class)
    OpsRegister()["RCNNDetOutputWithAttr"].kpts_classify_pad_ratio = kpts_param.kpts_classify_pad_ratio
    # parsing atrs_param pkg
    atrs_param = detect_output_ssd.atrs_param
    OpsRegister()["RCNNDetOutputWithAttr"].atrs_reg_bottom_idx = atrs_param.atrs_reg_bottom_idx
    OpsRegister()["RCNNDetOutputWithAttr"].atrs_reg_norm_idx_st = atrs_param.atrs_reg_norm_idx_st
    atrs_norm_type_vec = []
    for norm_type in list(atrs_param.atrs_norm_type):
        if norm_type == ATRSParameter.WIDTH:
            atrs_norm_type_vec.append("WIDTH")
        elif norm_type == ATRSParameter.HEIGHT:
            atrs_norm_type_vec.append("HEIGHT")
        elif norm_type == ATRSParameter.WIDTH_LOG:
            atrs_norm_type_vec.append("WIDTH_LOG")
        elif norm_type == ATRSParameter.HEIGHT_LOG:
            atrs_norm_type_vec.append("HEIGHT_LOG")
        else:
            atrs_norm_type_vec.append("NONE")
    OpsRegister()["RCNNDetOutputWithAttr"].atrs_norm_type = atrs_norm_type_vec
    # parsing ftrs_param pkg
    ftrs_param = detect_output_ssd.ftrs_param
    OpsRegister()["RCNNDetOutputWithAttr"].ftrs_bottom_idx = ftrs_param.ftrs_bottom_idx
    # parsing spmp_param pkg
    spmp_param = detect_output_ssd.spmp_param
    OpsRegister()["RCNNDetOutputWithAttr"].spmp_bottom_idx = spmp_param.spmp_bottom_idx
    OpsRegister()["RCNNDetOutputWithAttr"].spmp_class_aware = list(spmp_param.spmp_class_aware)
    OpsRegister()["RCNNDetOutputWithAttr"].spmp_label_width = list(spmp_param.spmp_label_width)
    OpsRegister()["RCNNDetOutputWithAttr"].spmp_label_height = list(spmp_param.spmp_label_height)
    OpsRegister()["RCNNDetOutputWithAttr"].spmp_pad_ratio = list(spmp_param.spmp_pad_ratio)
    # parsing cam3d_param pkg
    cam3d_param = detect_output_ssd.cam3d_param
    OpsRegister()["RCNNDetOutputWithAttr"].cam3d_bottom_idx = cam3d_param.cam3d_bottom_idx


@ParserFeedDecorator("DFMBPSROIAlign")
def Parser_dfmbps_roi_align(args):
    layer = args[1]
    # parser caffe parameter
    dfmb_psroi_pooling = layer.dfmb_psroi_pooling_param
    OpsRegister()["DFMBPSROIAlign"].heat_map_a = dfmb_psroi_pooling.heat_map_a
    OpsRegister()["DFMBPSROIAlign"].heat_map_b = dfmb_psroi_pooling.heat_map_b
    OpsRegister()["DFMBPSROIAlign"].pad_ratio = dfmb_psroi_pooling.pad_ratio
    OpsRegister()["DFMBPSROIAlign"].output_dim = dfmb_psroi_pooling.output_dim
    OpsRegister()["DFMBPSROIAlign"].trans_std = dfmb_psroi_pooling.trans_std
    OpsRegister()["DFMBPSROIAlign"].sample_per_part = dfmb_psroi_pooling.sample_per_part
    OpsRegister()["DFMBPSROIAlign"].group_height = dfmb_psroi_pooling.group_height
    OpsRegister()["DFMBPSROIAlign"].group_width = dfmb_psroi_pooling.group_width
    OpsRegister()["DFMBPSROIAlign"].pooled_height = dfmb_psroi_pooling.pooled_height
    OpsRegister()["DFMBPSROIAlign"].pooled_width = dfmb_psroi_pooling.pooled_width
    OpsRegister()["DFMBPSROIAlign"].part_height = dfmb_psroi_pooling.part_height
    OpsRegister()["DFMBPSROIAlign"].part_width = dfmb_psroi_pooling.part_width


@ParserFeedDecorator("RCNNProposal")
def Parser_rcnn_proposal(args):
    layer = args[1]
    # parser caffe parameter
    #OpsRegister()["Split"].split_num = len(top)
    # parser caffe parameter
    detect_output_ssd = layer.detection_output_ssd_param
    OpsRegister()["RCNNProposal"].threshold = list(detect_output_ssd.threshold)
    OpsRegister()["RCNNProposal"].channel_per_scale = detect_output_ssd.channel_per_scale
    OpsRegister()["RCNNProposal"].class_name_list = str(detect_output_ssd.class_name_list)
    OpsRegister()["RCNNProposal"].num_class = detect_output_ssd.num_class
    OpsRegister()["RCNNProposal"].refine_out_of_map_bbox = detect_output_ssd.refine_out_of_map_bbox
    OpsRegister()["RCNNProposal"].class_indexes = list(detect_output_ssd.class_indexes)
    OpsRegister()["RCNNProposal"].heat_map_a = list(detect_output_ssd.heat_map_a)
    OpsRegister()["RCNNProposal"].heat_map_b = list(detect_output_ssd.heat_map_b)
    OpsRegister()["RCNNProposal"].threshold_objectness = detect_output_ssd.threshold_objectness
    OpsRegister()["RCNNProposal"].proposal_min_sqrt_area = list(detect_output_ssd.proposal_min_sqrt_area)
    OpsRegister()["RCNNProposal"].proposal_max_sqrt_area = list(detect_output_ssd.proposal_max_sqrt_area)
    OpsRegister()["RCNNProposal"].bg_as_one_of_softmax = detect_output_ssd.bg_as_one_of_softmax
    OpsRegister()["RCNNProposal"].use_target_type_rcnn = detect_output_ssd.use_target_type_rcnn
    OpsRegister()["RCNNProposal"].im_width = detect_output_ssd.im_width
    OpsRegister()["RCNNProposal"].im_height = detect_output_ssd.im_height
    OpsRegister()["RCNNProposal"].rpn_proposal_output_score = detect_output_ssd.rpn_proposal_output_score
    OpsRegister()["RCNNProposal"].regress_agnostic = detect_output_ssd.regress_agnostic
    OpsRegister()["RCNNProposal"].allow_border = detect_output_ssd.allow_border
    OpsRegister()["RCNNProposal"].allow_border_ratio = detect_output_ssd.allow_border_ratio
    OpsRegister()["RCNNProposal"].bbox_size_add_one = detect_output_ssd.bbox_size_add_one
    OpsRegister()["RCNNProposal"].read_width_scale = detect_output_ssd.read_width_scale
    OpsRegister()["RCNNProposal"].read_height_scale = detect_output_ssd.read_height_scale
    OpsRegister()["RCNNProposal"].read_height_offset = detect_output_ssd.read_height_offset
    OpsRegister()["RCNNProposal"].min_size_h = detect_output_ssd.min_size_h
    OpsRegister()["RCNNProposal"].min_size_w = detect_output_ssd.min_size_w
    if detect_output_ssd.min_size_mode == DetectionOutputSSDParameter.HEIGHT_AND_WIDTH:
        OpsRegister()["RCNNProposal"].min_size_mode = "HEIGHT_AND_WIDTH"
    else:
        OpsRegister()["RCNNProposal"].min_size_mode = "HEIGHT_OR_WIDTH"
    # parsing nms_param pkg
    nms_param = detect_output_ssd.nms_param
    OpsRegister()["RCNNProposal"].need_nms = nms_param.need_nms
    OpsRegister()["RCNNProposal"].overlap_ratio = list(nms_param.overlap_ratio)
    OpsRegister()["RCNNProposal"].top_n = list(nms_param.top_n)
    OpsRegister()["RCNNProposal"].add_score = nms_param.add_score
    OpsRegister()["RCNNProposal"].max_candidate_n = list(nms_param.max_candidate_n)
    OpsRegister()["RCNNProposal"].use_soft_nms = list(nms_param.use_soft_nms)
    OpsRegister()["RCNNProposal"].nms_among_classes = nms_param.nms_among_classes
    OpsRegister()["RCNNProposal"].voting = list(nms_param.voting)
    OpsRegister()["RCNNProposal"].vote_iou = list(nms_param.vote_iou)
    OpsRegister()["RCNNProposal"].nms_gpu_max_n_per_time = nms_param.nms_gpu_max_n_per_time
    # parsing gen_anchor_param pkg
    gen_anchor_param = detect_output_ssd.gen_anchor_param
    OpsRegister()["RCNNProposal"].base_size = gen_anchor_param.base_size
    OpsRegister()["RCNNProposal"].ratios = list(gen_anchor_param.ratios)
    OpsRegister()["RCNNProposal"].scales = list(gen_anchor_param.scales)
    OpsRegister()["RCNNProposal"].anchor_width = list(gen_anchor_param.anchor_width)
    OpsRegister()["RCNNProposal"].anchor_height = list(gen_anchor_param.anchor_height)
    OpsRegister()["RCNNProposal"].anchor_x1 = list(gen_anchor_param.anchor_x1)
    OpsRegister()["RCNNProposal"].anchor_y1 = list(gen_anchor_param.anchor_y1)
    OpsRegister()["RCNNProposal"].anchor_x2 = list(gen_anchor_param.anchor_x2)
    OpsRegister()["RCNNProposal"].anchor_y2 = list(gen_anchor_param.anchor_y2)
    OpsRegister()["RCNNProposal"].zero_anchor_center = gen_anchor_param.zero_anchor_center
    # parsing kpts_param pkg
    kpts_param = detect_output_ssd.kpts_param
    OpsRegister()["RCNNProposal"].kpts_exist_bottom_idx = kpts_param.kpts_exist_bottom_idx
    OpsRegister()["RCNNProposal"].kpts_reg_bottom_idx = kpts_param.kpts_reg_bottom_idx
    OpsRegister()["RCNNProposal"].kpts_reg_as_classify = kpts_param.kpts_reg_as_classify
    OpsRegister()["RCNNProposal"].kpts_classify_width = kpts_param.kpts_classify_width
    OpsRegister()["RCNNProposal"].kpts_classify_height = kpts_param.kpts_classify_height
    OpsRegister()["RCNNProposal"].kpts_reg_norm_idx_st = kpts_param.kpts_reg_norm_idx_st
    OpsRegister()["RCNNProposal"].kpts_st_for_each_class = list(kpts_param.kpts_st_for_each_class)
    OpsRegister()["RCNNProposal"].kpts_ed_for_each_class = list(kpts_param.kpts_ed_for_each_class)
    OpsRegister()["RCNNProposal"].kpts_classify_pad_ratio = kpts_param.kpts_classify_pad_ratio
    # parsing atrs_param pkg
    atrs_param = detect_output_ssd.atrs_param
    OpsRegister()["RCNNProposal"].atrs_reg_bottom_idx = atrs_param.atrs_reg_bottom_idx
    OpsRegister()["RCNNProposal"].atrs_reg_norm_idx_st = atrs_param.atrs_reg_norm_idx_st
    atrs_norm_type_vec = []
    for norm_type in list(atrs_param.atrs_norm_type):
        if norm_type == ATRSParameter.WIDTH:
            atrs_norm_type_vec.append("WIDTH")
        elif norm_type == ATRSParameter.HEIGHT:
            atrs_norm_type_vec.append("HEIGHT")
        elif norm_type == ATRSParameter.WIDTH_LOG:
            atrs_norm_type_vec.append("WIDTH_LOG")
        elif norm_type == ATRSParameter.HEIGHT_LOG:
            atrs_norm_type_vec.append("HEIGHT_LOG")
        else:
            atrs_norm_type_vec.append("NONE")
    OpsRegister()["RCNNProposal"].atrs_norm_type = atrs_norm_type_vec
    # parsing ftrs_param pkg
    ftrs_param = detect_output_ssd.ftrs_param
    OpsRegister()["RCNNProposal"].ftrs_bottom_idx = ftrs_param.ftrs_bottom_idx
    # parsing spmp_param pkg
    spmp_param = detect_output_ssd.spmp_param
    OpsRegister()["RCNNProposal"].spmp_bottom_idx = spmp_param.spmp_bottom_idx
    OpsRegister()["RCNNProposal"].spmp_class_aware = list(spmp_param.spmp_class_aware)
    OpsRegister()["RCNNProposal"].spmp_label_width = list(spmp_param.spmp_label_width)
    OpsRegister()["RCNNProposal"].spmp_label_height = list(spmp_param.spmp_label_height)
    OpsRegister()["RCNNProposal"].spmp_pad_ratio = list(spmp_param.spmp_pad_ratio)
    # parsing cam3d_param pkg
    cam3d_param = detect_output_ssd.cam3d_param
    OpsRegister()["RCNNProposal"].cam3d_bottom_idx = cam3d_param.cam3d_bottom_idx

    bbox_reg = layer.bbox_reg_param
    OpsRegister()["RCNNProposal"].bbox_mean = list(bbox_reg.bbox_mean)
    OpsRegister()["RCNNProposal"].bbox_std = list(bbox_reg.bbox_std)


@ParserFeedDecorator("ProposalImgScaleToCamCoords")
def Parser_proposal_img_scale_to_cam_coords(args):
    layer = args[1]
    # parser caffe parameter
    proposal_img_scale_to_cam_coords = layer.proposal_img_scale_to_cam_coords_param
    OpsRegister()["ProposalImgScaleToCamCoords"].num_class = proposal_img_scale_to_cam_coords.num_class
    OpsRegister()["ProposalImgScaleToCamCoords"].sub_class_num_class = list(proposal_img_scale_to_cam_coords.sub_class_num_class)
    OpsRegister()["ProposalImgScaleToCamCoords"].sub_class_bottom_idx = list(proposal_img_scale_to_cam_coords.sub_class_bottom_idx)
    if proposal_img_scale_to_cam_coords.prj_h_norm_type == ProposalImgScaleToCamCoordsParameter.HEIGHT:
        OpsRegister()["ProposalImgScaleToCamCoords"].prj_h_norm_type = "HEIGHT"
    else:
        OpsRegister()["ProposalImgScaleToCamCoords"].prj_h_norm_type = "HEIGHT_LOG"
    OpsRegister()["ProposalImgScaleToCamCoords"].has_size3d_and_orien3d = proposal_img_scale_to_cam_coords.has_size3d_and_orien3d
    if proposal_img_scale_to_cam_coords.orien_type == ProposalImgScaleToCamCoordsParameter.PI:
        OpsRegister()["ProposalImgScaleToCamCoords"].orien_type = "PI"
    else:
        OpsRegister()["ProposalImgScaleToCamCoords"].orien_type = "PI2"
    OpsRegister()["ProposalImgScaleToCamCoords"].cls_ids_zero_size3d_w = list(proposal_img_scale_to_cam_coords.cls_ids_zero_size3d_w)
    OpsRegister()["ProposalImgScaleToCamCoords"].cls_ids_zero_size3d_l = list(proposal_img_scale_to_cam_coords.cls_ids_zero_size3d_l)
    OpsRegister()["ProposalImgScaleToCamCoords"].cls_ids_zero_orien3d = list(proposal_img_scale_to_cam_coords.cls_ids_zero_orien3d)
    OpsRegister()["ProposalImgScaleToCamCoords"].cmp_pts_corner_3d = proposal_img_scale_to_cam_coords.cmp_pts_corner_3d
    OpsRegister()["ProposalImgScaleToCamCoords"].cmp_pts_corner_2d = proposal_img_scale_to_cam_coords.cmp_pts_corner_2d
    OpsRegister()["ProposalImgScaleToCamCoords"].ctr_2d_means = list(proposal_img_scale_to_cam_coords.ctr_2d_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].ctr_2d_stds = list(proposal_img_scale_to_cam_coords.ctr_2d_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].prj_h_means = list(proposal_img_scale_to_cam_coords.prj_h_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].prj_h_stds = list(proposal_img_scale_to_cam_coords.prj_h_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_h_means = list(proposal_img_scale_to_cam_coords.real_h_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_h_stds = list(proposal_img_scale_to_cam_coords.real_h_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_w_means = list(proposal_img_scale_to_cam_coords.real_w_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_w_stds = list(proposal_img_scale_to_cam_coords.real_w_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_l_means = list(proposal_img_scale_to_cam_coords.real_l_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_l_stds = list(proposal_img_scale_to_cam_coords.real_l_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].sin_means = list(proposal_img_scale_to_cam_coords.sin_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].sin_stds = list(proposal_img_scale_to_cam_coords.sin_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].cos_means = list(proposal_img_scale_to_cam_coords.cos_means)
    OpsRegister()["ProposalImgScaleToCamCoords"].cos_stds = list(proposal_img_scale_to_cam_coords.cos_stds)
    OpsRegister()["ProposalImgScaleToCamCoords"].cam_info_idx_st_in_im_info = proposal_img_scale_to_cam_coords.cam_info_idx_st_in_im_info
    OpsRegister()["ProposalImgScaleToCamCoords"].im_width_scale = proposal_img_scale_to_cam_coords.im_width_scale
    OpsRegister()["ProposalImgScaleToCamCoords"].im_height_scale = proposal_img_scale_to_cam_coords.im_height_scale
    OpsRegister()["ProposalImgScaleToCamCoords"].cords_offset_x = proposal_img_scale_to_cam_coords.cords_offset_x
    OpsRegister()["ProposalImgScaleToCamCoords"].cords_offset_y = proposal_img_scale_to_cam_coords.cords_offset_y
    OpsRegister()["ProposalImgScaleToCamCoords"].bbox_size_add_one = proposal_img_scale_to_cam_coords.bbox_size_add_one
    OpsRegister()["ProposalImgScaleToCamCoords"].rotate_coords_by_pitch = proposal_img_scale_to_cam_coords.rotate_coords_by_pitch
    #OpsRegister()["ProposalImgScaleToCamCoords"].refine_coords_by_bbox = proposal_img_scale_to_cam_coords.refine_coords_by_bbox
    #OpsRegister()["ProposalImgScaleToCamCoords"].refine_min_dist = proposal_img_scale_to_cam_coords.refine_min_dist
    #OpsRegister()["ProposalImgScaleToCamCoords"].refine_dist_for_height_ratio_one = proposal_img_scale_to_cam_coords.refine_dist_for_height_ratio_one
    #OpsRegister()["ProposalImgScaleToCamCoords"].max_3d2d_height_ratio_for_min_dist = proposal_img_scale_to_cam_coords.max_3d2d_height_ratio_for_min_dist
    OpsRegister()["ProposalImgScaleToCamCoords"].with_trunc_ratio = proposal_img_scale_to_cam_coords.with_trunc_ratio
    OpsRegister()["ProposalImgScaleToCamCoords"].regress_ph_rh_as_whole = proposal_img_scale_to_cam_coords.regress_ph_rh_as_whole
    OpsRegister()["ProposalImgScaleToCamCoords"].real_h_means_as_whole = list(proposal_img_scale_to_cam_coords.real_h_means_as_whole)
    OpsRegister()["ProposalImgScaleToCamCoords"].real_h_stds_as_whole = list(proposal_img_scale_to_cam_coords.real_h_stds_as_whole)

@ParserFeedDecorator("RoisAnchorFeature")
def Parser_rois_anchor_feature(args):
    layer = args[1]
    # parser caffe parameter
    rois_anchor_feature_param = layer.rois_anchor_feature_param
    OpsRegister()["RoisAnchorFeature"].min_anchor_size = rois_anchor_feature_param.min_anchor_size
    OpsRegister()["RoisAnchorFeature"].num_anchor_scales = rois_anchor_feature_param.num_anchor_scales
    OpsRegister()["RoisAnchorFeature"].anchor_scale_pow_base = rois_anchor_feature_param.anchor_scale_pow_base
    OpsRegister()["RoisAnchorFeature"].anchor_wph_ratios = list(rois_anchor_feature_param.anchor_wph_ratios)
    OpsRegister()["RoisAnchorFeature"].num_top_iou_anchor = rois_anchor_feature_param.num_top_iou_anchor
    OpsRegister()["RoisAnchorFeature"].min_num_top_iou_anchor = rois_anchor_feature_param.min_num_top_iou_anchor
    OpsRegister()["RoisAnchorFeature"].iou_thr = rois_anchor_feature_param.iou_thr
    OpsRegister()["RoisAnchorFeature"].ft_ratio_h = rois_anchor_feature_param.ft_ratio_h
    OpsRegister()["RoisAnchorFeature"].ft_ratio_w = rois_anchor_feature_param.ft_ratio_w
    OpsRegister()["RoisAnchorFeature"].ft_log_ratio_h = rois_anchor_feature_param.ft_log_ratio_h
    OpsRegister()["RoisAnchorFeature"].ft_log_ratio_w = rois_anchor_feature_param.ft_log_ratio_w
    OpsRegister()["RoisAnchorFeature"].bbox_size_add_one = rois_anchor_feature_param.bbox_size_add_one

@ParserFeedDecorator("Axpy")
def Parser_axpy(args):
    pass


@ParserFeedDecorator("PriorBox")
def Parser_priorbox(args):
    layer = args[1]
    prior_box_param = layer.prior_box_param
    if len(prior_box_param.min_size) or len(prior_box_param.max_size) or \
    len(prior_box_param.aspect_ratio):
        OpsRegister()["PriorBox"].min_size = list(prior_box_param.min_size)
        OpsRegister()["PriorBox"].max_size = list(prior_box_param.max_size)
        OpsRegister()["PriorBox"].aspect_ratio = list(prior_box_param.aspect_ratio)
    elif len(prior_box_param.fixed_size) or len(prior_box_param.fixed_ratio) or \
    len(prior_box_param.density):
        OpsRegister()["PriorBox"].fixed_size = list(prior_box_param.fixed_size)
        OpsRegister()["PriorBox"].fixed_ratio = list(prior_box_param.fixed_ratio)
        density_list = list(prior_box_param.density)
        OpsRegister()["PriorBox"].density = map(float, density_list)
    OpsRegister()["PriorBox"].is_flip = prior_box_param.flip
    OpsRegister()["PriorBox"].is_clip = prior_box_param.clip
    OpsRegister()["PriorBox"].variance = list(prior_box_param.variance)
    OpsRegister()["PriorBox"].img_h = prior_box_param.img_h
    OpsRegister()["PriorBox"].img_w = prior_box_param.img_w
    if prior_box_param.HasField('step_h') and pooling_param.HasField('step_w'):
        OpsRegister()["PriorBox"].step_h = prior_box_param.step_h
        OpsRegister()["PriorBox"].step_w = prior_box_param.step_w
    elif prior_box_param.HasField('step'):
        OpsRegister()["PriorBox"].step_h = prior_box_param.step
        OpsRegister()["PriorBox"].step_w = prior_box_param.step
    OpsRegister()["PriorBox"].offset = prior_box_param.offset
    OpsRegister()["PriorBox"].order = ['MIN', 'MAX', 'COM']


@ParserFeedDecorator("DetectionOutput")
def Parser_detectionoutput(args):
    layer = args[1]
    detection_output_param = layer.detection_output_param
    nms_param = detection_output_param.nms_param
    OpsRegister()["DetectionOutput"].share_location = detection_output_param.share_location
    OpsRegister()["DetectionOutput"].variance_encode_in_target = detection_output_param.variance_encoded_in_target
    OpsRegister()["DetectionOutput"].class_num = detection_output_param.num_classes
    OpsRegister()["DetectionOutput"].background_id = detection_output_param.background_label_id
    OpsRegister()["DetectionOutput"].keep_top_k = detection_output_param.keep_top_k
    OpsRegister()["DetectionOutput"].conf_thresh = detection_output_param.confidence_threshold
    OpsRegister()["DetectionOutput"].nms_top_k = nms_param.top_k
    OpsRegister()["DetectionOutput"].nms_thresh = nms_param.nms_threshold
    OpsRegister()["DetectionOutput"].nms_eta = nms_param.eta
    code_type = ""
    if detection_output_param.code_type == PriorBoxParameter.CORNER:
        code_type = "CORNER"
    elif detection_output_param.code_type == PriorBoxParameter.CENTER_SIZE:
        code_type = "CENTER_SIZE"
    elif detection_output_param.code_type == PriorBoxParameter.CORNER_SIZE:
        code_type = "CORNER_SIZE"
    OpsRegister()["DetectionOutput"].code_type = code_type


@ParserFeedDecorator("Argmax")
def Parser_argmax(args):
    layer = args[1]
    argmax_param = layer.argmax_param
    OpsRegister()["Argmax"].out_max_val = argmax_param.out_max_val
    OpsRegister()["Argmax"].top_k = argmax_param.top_k
    OpsRegister()["Argmax"].axis = argmax_param.axis
    OpsRegister()["Argmax"].axis_term = True


@ParserFeedDecorator("Normalize")
def Parser_normalize(args):
    layer = args[1]
    if len(args) == 4:
        norm_param = layer.norm_param
        scale_filler = norm_param.scale_filler
        OpsRegister()["Normalize"].begin_norm_axis = -1
        OpsRegister()["Normalize"].is_across_spatial = norm_param.across_spatial
        OpsRegister()["Normalize"].is_shared_channel = norm_param.channel_shared
        OpsRegister()["Normalize"].eps = norm_param.eps
        OpsRegister()["Normalize"].p = 2
    else:
        private_data = args[4]
        assert private_data['use_global_stats'] is True
        OpsRegister()["Normalize"].begin_norm_axis = -1
        OpsRegister()["Normalize"].is_across_spatial = False
        OpsRegister()["Normalize"].is_shared_channel = False
        OpsRegister()["Normalize"].eps = 1e-6
        OpsRegister()["Normalize"].p = 2

@ParserFeedDecorator("Activation")
def Parser_relu6(args):
    layer = args[1]
    relu6_param = layer.relu6_param
    OpsRegister()["Activation"].type = "ClippedRelu"
    OpsRegister()["Activation"].clip_relu_num = 6

@ParserFeedDecorator("Interp")
def Parser_interp(args):
    layer = args[1]
    interp_param = layer.interp_param
    OpsRegister()["Interp"].height = interp_param.height
    OpsRegister()["Interp"].width = interp_param.width
    OpsRegister()["Interp"].zoom_factor = interp_param.zoom_factor
    OpsRegister()["Interp"].shrink_factor = interp_param.shrink_factor
    OpsRegister()["Interp"].pad_beg = interp_param.pad_beg
    OpsRegister()["Interp"].pad_end = interp_param.pad_end

# caffe layer parameter parser map
CAFFE_LAYER_PARSER = {
                "Split": OpsParam().set_parser(Parser_split),
                "Accuracy": OpsParam().set_parser(NotNeededInInference),
                "ArgMax": OpsParam().set_parser(NotNeededInInference),
                "BatchNorm": OpsParam().set_parser(Parser_batch_norm),
                "Bias": OpsParam().set_parser(NotNeededInInference),
                "Concat": OpsParam().set_parser(Parser_concat),
                "ContrastiveLoss": OpsParam().set_parser(NotNeededInInference),
                "Convolution": OpsParam().set_parser(Parser_convolution),
                "ConvolutionDepthwise": OpsParam().set_parser(Parser_convolutiondepthwise),
                "DepthwiseConvolution": OpsParam().set_parser(Parser_convolutiondepthwise),
                "Deconvolution": OpsParam().set_parser(Parser_deconvolution),
                "DeformableConvolution": OpsParam().set_parser(Parser_deformable_convolution),
                "Crop": OpsParam().set_parser(Parser_crop),
                "Data": OpsParam().set_parser(NotNeededInInference),
                "Dropout": OpsParam().set_parser(Parser_dropout),
                "DummyData": OpsParam().set_parser(NotNeededInInference),
                "Eltwise": OpsParam().set_parser(Parser_eltwise),
                "ELU": OpsParam().set_parser(Parser_elu),
                "Embed": OpsParam().set_parser(Parser_embed),
                "Exp": OpsParam().set_parser(Parser_exp),
                "Flatten": OpsParam().set_parser(Parser_flatten),
                "HDF5Data": OpsParam().set_parser(NotNeededInInference),
                "HDF5Output": OpsParam().set_parser(NotNeededInInference),
                "HingeLoss": OpsParam().set_parser(NotNeededInInference),
                "ImageData": OpsParam().set_parser(NotNeededInInference),
                "InfogainLoss": OpsParam().set_parser(NotNeededInInference),
                "InnerProduct": OpsParam().set_parser(Parser_innerproduct),
                "Input": OpsParam().set_parser(Parser_input),
                "Log": OpsParam().set_parser(Parser_log),
                "LRN": OpsParam().set_parser(Parser_lrn),
                "MemoryData": OpsParam().set_parser(NotNeededInInference),
                "MVN": OpsParam().set_parser(Parser_mvn),
                "Parameter": OpsParam().set_parser(NotNeededInInference),
                "Pooling": OpsParam().set_parser(Parser_pooling),
                "Power": OpsParam().set_parser(Parser_power),
                "PReLU": OpsParam().set_parser(Parser_prelu),
                "Permute": OpsParam().set_parser(Parser_permute),
                "Python": OpsParam().set_parser(NotNeededInInference),
                "Recurrent": OpsParam().set_parser(Parser_rnn_ori),
                "RNN": OpsParam().set_parser(Parser_rnn_ori),
                "LSTM": OpsParam().set_parser(Parser_rnn_lstm),
                "Reduction": OpsParam().set_parser(NotNeededInInference),
                "ReLU": OpsParam().set_parser(Parser_relu),
                "Reshape": OpsParam().set_parser(Parser_reshape),
                "Scale": OpsParam().set_parser(Parser_scale),
                "Sigmoid": OpsParam().set_parser(Parser_sigmoid),
                "Softmax": OpsParam().set_parser(Parser_softmax),
                "SPP": OpsParam().set_parser(Parser_spp),
                "Slice": OpsParam().set_parser(Parser_slice),
                "TanH": OpsParam().set_parser(Parser_tanh),
                "Threshold": OpsParam().set_parser(NotNeededInInference),
                "Tile": OpsParam().set_parser(NotNeededInInference),
                "WindowData": OpsParam().set_parser(NotNeededInInference),
                "RPNProposalSSD": OpsParam().set_parser(Parser_rpn_proposal_ssd), # adu add
                "RCNNDetOutputWithAttr": OpsParam().set_parser(Parser_rcnn_net_output_with_attr), # adu add
                "DFMBPSROIAlign": OpsParam().set_parser(Parser_dfmbps_roi_align), # adu add
                "RCNNProposal": OpsParam().set_parser(Parser_rcnn_proposal), # adu add
                "ProposalImgScaleToCamCoords": OpsParam().set_parser(Parser_proposal_img_scale_to_cam_coords), # adu add
                "Axpy": OpsParam().set_parser(Parser_axpy), # vis add
                "PriorBox": OpsParam().set_parser(Parser_priorbox), # vis add
                "DetectionOutput": OpsParam().set_parser(Parser_detectionoutput), # vis add
                "ArgMax": OpsParam().set_parser(Parser_argmax),
                "Normalize": OpsParam().set_parser(Parser_normalize),
                "Resize": OpsParam().set_parser(Parser_resize),
                "ReLU6": OpsParam().set_parser(Parser_relu6),
                "Normalization": OpsParam().set_parser(Parser_normalize),
                "ShuffleChannel": OpsParam().set_parser(Parser_ShuffleChannel),
                "Coord2Patch": OpsParam().set_parser(Parser_Coord2Patch),
                "RoisAnchorFeature": OpsParam().set_parser(Parser_rois_anchor_feature),
                "Interp": OpsParam().set_parser(Parser_interp)
                }
