from ..operations import OpsParam, OpsRegister
from ..logger import *
from ..proto import *
from fluid_helper import *


def ParserFeedDecorator(OpName):
    def warpper(Parser):
        def warpper_args(args):
            Parser(args)
            OpsRegister()[OpName].feed_node_attr(args[0])
            args[2].set_name(OpName)
            args[0].set_op(args[2]())
        return warpper_args
    return warpper

# common 
def NotNeededInInference(args):
    # args is tuple object
    node_io = args[0]
    layer = args[1]

@ParserFeedDecorator("Input")
def Parser_feed(args):
    private_data = args[4]
    input_shape = private_data['input_shape']
    alias = private_data['alias']
    OpsRegister()["Input"].input_shape = input_shape
    OpsRegister()["Input"].alias = alias

@ParserFeedDecorator("Convolution")
def Parser_conv2d(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter')
    OpsRegister()["Convolution"].weight_1 = weights_tensor
    OpsRegister()["Convolution"].filter_num = weights_shape[0]
    OpsRegister()["Convolution"].kernel_size = weights_shape[-2:]
    OpsRegister()["Convolution"].strides = helper.attr_data(op, 'strides')
    OpsRegister()["Convolution"].padding = helper.attr_data(op, 'paddings')
    OpsRegister()["Convolution"].dilation_rate = helper.attr_data(op, 'dilations')
    OpsRegister()["Convolution"].group = helper.attr_data(op, 'groups')
    OpsRegister()["Convolution"].axis = 1
    if 'bias' in private_data.keys():
        OpsRegister()["Convolution"].bias_term = True
        OpsRegister()["Convolution"].weight_2 = private_data['bias']
    else:
        OpsRegister()["Convolution"].bias_term = False

@ParserFeedDecorator("Deconvolution")
def Parser_conv2d_transpose(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter')
    weights_tensor.set_shape([weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3]])
    OpsRegister()["Deconvolution"].weight_1 = weights_tensor
    OpsRegister()["Deconvolution"].filter_num = weights_shape[1]
    OpsRegister()["Deconvolution"].kernel_size = weights_shape[-2:]
    OpsRegister()["Deconvolution"].strides = helper.attr_data(op, 'strides')
    OpsRegister()["Deconvolution"].padding = helper.attr_data(op, 'paddings')
    OpsRegister()["Deconvolution"].dilation_rate = helper.attr_data(op, 'dilations')
    OpsRegister()["Deconvolution"].group = helper.attr_data(op, 'groups')
    OpsRegister()["Deconvolution"].axis = 1
    if 'bias' in private_data.keys():
        OpsRegister()["Deconvolution"].bias_term = True
        OpsRegister()["Deconvolution"].weight_2 = private_data['bias']
    else:
        OpsRegister()["Deconvolution"].bias_term = False

@ParserFeedDecorator("ReLU")
def Parser_relu(args):
    OpsRegister()["ReLU"].alpha = 0.0

@ParserFeedDecorator("Pooling")
def Parser_pool2d(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Pooling"].pool_size = helper.attr_data(op, 'ksize')
    OpsRegister()["Pooling"].strides = helper.attr_data(op, 'strides')
    OpsRegister()["Pooling"].padding = helper.attr_data(op, 'paddings')
    OpsRegister()["Pooling"].global_pooling = helper.attr_data(op, 'global_pooling')
    if helper.attr_data(op, 'pooling_type') == 'max':
        OpsRegister()["Pooling"].method = "MAX"
    elif helper.attr_data(op, 'pooling_type') in ['average', 'avg']:
        OpsRegister()["Pooling"].method = "AVG"
    if helper.attr_data(op, 'ceil_mode') == False:
        OpsRegister()["Pooling"].cmp_out_shape_floor_as_conv = True
    else:
        OpsRegister()["Pooling"].cmp_out_shape_floor_as_conv = False

@ParserFeedDecorator("Dense")
def Parser_mul(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    weights_needs_trans = True
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Y', weights_needs_trans)
    OpsRegister()["Dense"].weight_1 = weights_tensor
    OpsRegister()["Dense"].out_dim = weights_shape[2]
    OpsRegister()["Dense"].axis = helper.attr_data(op, 'x_num_col_dims')
    if 'bias' in private_data.keys():
        OpsRegister()["Dense"].bias_term = True
        OpsRegister()["Dense"].weight_2 = private_data['bias']
    else:
        OpsRegister()["Dense"].bias_term = False

@ParserFeedDecorator("Softmax")
def Parser_softmax(args):
    private_data = args[4]
    if 'axis' in private_data.keys():
        axis = private_data['axis']
    else:
        axis = 1
    OpsRegister()["Softmax"].axis = axis

@ParserFeedDecorator("Activation")
def Parser_sigmoid(args):
    OpsRegister()["Activation"].type = "Sigmoid"

@ParserFeedDecorator("Axpy")
def Parser_axpy(args):
    pass

@ParserFeedDecorator("BatchNorm")
def Parser_batch_norm(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["BatchNorm"].weight_1 = helper.param_tensor(op, 'Mean')
    OpsRegister()["BatchNorm"].weight_2 = helper.param_tensor(op, 'Variance')
    OpsRegister()["BatchNorm"].weight_3 = helper.create_tensor([1], [1, 1, 1, 1], FLOAT)
    OpsRegister()["BatchNorm"].momentum = helper.attr_data(op, 'momentum')
    OpsRegister()["BatchNorm"].epsilon = helper.attr_data(op, 'epsilon')

@ParserFeedDecorator("Scale")
def Parser_scale_disc_bn(args):
    op = args[1]
    helper = args[3]
    mean = helper.np_param(op, 'Mean')
    var = helper.np_param(op, 'Variance')
    alpha = helper.np_param(op, 'Scale')
    beta = helper.np_param(op, 'Bias')
    eps = helper.attr_data(op, 'epsilon')
    var = np.sqrt(var + eps)
    np_scale = alpha / var
    np_bias = beta - (alpha * mean / var)
    np_scale_shape = map(int, [1] * (4 - len(np_scale.shape)) + list(np_scale.shape))
    np_bias_shape = map(int, [1] * (4 - len(np_bias.shape)) + list(np_bias.shape))
    np_scale_tensor = helper.create_tensor(np_scale.flatten().tolist(), np_scale_shape, FLOAT)
    np_bias_tensor = helper.create_tensor(np_bias.flatten().tolist(), np_bias_shape, FLOAT)
    OpsRegister()["Scale"].bias_term = True
    OpsRegister()["Scale"].weight_1 = np_scale_tensor
    OpsRegister()["Scale"].weight_2 = np_bias_tensor
    OpsRegister()["Scale"].axis = 1
    OpsRegister()["Scale"].num_axes = 1

@ParserFeedDecorator("Scale")
def Parser_scale_of_bn(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Scale"].weight_1 = helper.param_tensor(op, 'Scale')
    OpsRegister()["Scale"].axis = 1
    OpsRegister()["Scale"].num_axes = 1
    has_bias = helper.is_persistable_param(op, 'Bias')
    if has_bias is True:
        OpsRegister()["Scale"].bias_term = True
        OpsRegister()["Scale"].weight_2 = helper.param_tensor(op, 'Bias')
    else:
        OpsRegister()["Scale"].bias_term = False

@ParserFeedDecorator("Split")
def Parser_split_ins(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    if 'split_num' in private_data.keys():
        split_num = private_data['split_num']
        OpsRegister()["Split"].split_num = split_num
    else:
        raise NameError('ERROR: Unknown Split_ins type.')

@ParserFeedDecorator("Slice")
def Parser_slice(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Slice"].slice_point = [-1]
    OpsRegister()["Slice"].num = helper.attr_data(op, 'num')
    OpsRegister()["Slice"].axis = helper.attr_data(op, 'axis')
    OpsRegister()["Slice"].sections = helper.attr_data(op, 'sections')

@ParserFeedDecorator("Reshape")
def Parser_reshape(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    layout = str()
    if 'new_shape' in private_data.keys():
        shape = private_data['new_shape']
    else:
        shape = helper.attr_data(op, 'shape')
    if len(shape) == 4:
        layout = 'NCHW'
    elif len(shape) == 3:
        layout = 'NHW'
    OpsRegister()["Reshape"].dims = shape
    OpsRegister()["Reshape"].layout = layout

@ParserFeedDecorator("Concat")
def Parser_concat(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Concat"].axis = helper.attr_data(op, 'axis')

@ParserFeedDecorator("Concat")
def Parser_concat_btw_priorbox_boxcoder(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    OpsRegister()["Concat"].axis = private_data['axis']

@ParserFeedDecorator("Permute")
def Parser_transpose(args):
    op = args[1]
    helper = args[3]
    fluid_dims = helper.attr_data(op, 'axis')
    n = 4 - len(fluid_dims)
    dims = range(0, n)
    tail_dims = [i + n for i in fluid_dims]
    dims.extend(tail_dims)
    OpsRegister()["Permute"].dims = dims


########## SSD Model ##########

@ParserFeedDecorator("PriorBox")
def Parser_prior_box(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["PriorBox"].min_size = helper.attr_data(op, 'min_sizes')
    OpsRegister()["PriorBox"].max_size = helper.attr_data(op, 'max_sizes')
    OpsRegister()["PriorBox"].aspect_ratio = helper.attr_data(op, 'aspect_ratios')
    OpsRegister()["PriorBox"].is_flip = helper.attr_data(op, 'flip')
    OpsRegister()["PriorBox"].is_clip = helper.attr_data(op, 'clip')
    OpsRegister()["PriorBox"].variance = helper.attr_data(op, 'variances')
    OpsRegister()["PriorBox"].img_h = 0
    OpsRegister()["PriorBox"].img_w = 0
    OpsRegister()["PriorBox"].step_h = helper.attr_data(op, 'step_h')
    OpsRegister()["PriorBox"].step_w = helper.attr_data(op, 'step_w')
    OpsRegister()["PriorBox"].offset = helper.attr_data(op, 'offset')
    OpsRegister()["PriorBox"].order = ['MIN', 'COM', 'MAX']

@ParserFeedDecorator("box_coder")
def Parser_box_coder(args):
    pass

@ParserFeedDecorator("DetectionOutput")
def Parser_multiclass_nms(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    OpsRegister()["DetectionOutput"].share_location = True
    OpsRegister()["DetectionOutput"].variance_encode_in_target = False
    OpsRegister()["DetectionOutput"].class_num = 0
    OpsRegister()["DetectionOutput"].background_id = helper.attr_data(op, 'background_label')
    OpsRegister()["DetectionOutput"].keep_top_k = helper.attr_data(op, 'keep_top_k')
    OpsRegister()["DetectionOutput"].conf_thresh = helper.attr_data(op, 'score_threshold')
    OpsRegister()["DetectionOutput"].nms_top_k = helper.attr_data(op, 'nms_top_k')
    OpsRegister()["DetectionOutput"].nms_thresh = helper.attr_data(op, 'nms_threshold')
    OpsRegister()["DetectionOutput"].nms_eta = helper.attr_data(op, 'nms_eta')
    if 'code_type' in private_data.keys():
        if private_data['code_type'] == 'decode_center_size':
            OpsRegister()["DetectionOutput"].code_type = "CENTER_SIZE"
    else:
        OpsRegister()["DetectionOutput"].code_type = "CORNER"


########## VIS Model ##########

@ParserFeedDecorator("Im2Sequence")
def Parser_im2sequence(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Im2Sequence"].paddings = helper.attr_data(op, 'paddings')
    OpsRegister()["Im2Sequence"].strides = helper.attr_data(op, 'strides')
    OpsRegister()["Im2Sequence"].window_size = helper.attr_data(op, 'kernels')
    OpsRegister()["Im2Sequence"].dilations = helper.attr_data(op, 'dilations', [1, 1])

@ParserFeedDecorator("Cast")
def Parser_cast(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Cast"].in_type = helper.attr_data(op, 'in_dtype')
    OpsRegister()["Cast"].out_type = helper.attr_data(op, 'out_dtype')

@ParserFeedDecorator("Argmax") # new256
def Parser_top_k(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Argmax"].out_max_val = True
    OpsRegister()["Argmax"].top_k = helper.attr_data(op, 'k')
    OpsRegister()["Argmax"].axis_term = False

@ParserFeedDecorator("CtcAlign")
def Parser_ctc_align(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["CtcAlign"].merge_repeated = helper.attr_data(op, 'merge_repeated')
    OpsRegister()["CtcAlign"].blank = helper.attr_data(op, 'blank')

@ParserFeedDecorator("Eltwise")
def Parser_sum(args):
    OpsRegister()["Eltwise"].type = "Add"
    OpsRegister()["Eltwise"].coeff = [1.0, 1.0]

@ParserFeedDecorator("LRN")
def Parser_lrn(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["LRN"].local_size = helper.attr_data(op, 'n')
    OpsRegister()["LRN"].alpha = helper.attr_data(op, 'alpha')
    OpsRegister()["LRN"].beta = helper.attr_data(op, 'beta')
    OpsRegister()["LRN"].norm_region = "ACROSS_CHANNELS"
    OpsRegister()["LRN"].k = helper.attr_data(op, 'k')

@ParserFeedDecorator("Gru")
def Parser_gru(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    OpsRegister()["Gru"].is_reverse = helper.attr_data(op, 'is_reverse')
    OpsRegister()["Gru"].gate_activation = helper.attr_data(op, 'gate_activation') + '_fluid'
    OpsRegister()["Gru"].activation = helper.attr_data(op, 'activation') + '_fluid'
    OpsRegister()["Gru"].gru_formula = "gru_origin"
    if bool(private_data) is True:
        ori_bx = private_data['np_bias_x']
        ori_bh = helper.np_param(op, 'Bias')
        ori_b = ori_bx + ori_bh
        ori_wx = private_data['np_weight_x']
        ori_wh = helper.np_param(op, 'Weight')
        new_tensors = helper.gru_tensor_convert(ori_wh, ori_wx, ori_b)
        weights = []
        for tensor in new_tensors:
            weights.append(helper.create_tensor(tensor.flatten().tolist(), \
                list(np.shape(tensor)), FLOAT))
        OpsRegister()["Gru"].weight_1 = weights[0]
        OpsRegister()["Gru"].weight_2 = weights[1]
    else:
        OpsRegister()["Gru"].weight_1 = helper.param_tensor(op, 'Weight')
        OpsRegister()["Gru"].weight_2 = helper.create_tensor([0], [-1], FLOAT)

@ParserFeedDecorator("LSTM")
def Parser_lstm(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    OpsRegister()["LSTM"].candidate_activation = helper.attr_data(op, 'candidate_activation')
    OpsRegister()["LSTM"].cell_activation = helper.attr_data(op, 'cell_activation')
    OpsRegister()["LSTM"].gate_activation = helper.attr_data(op, 'gate_activation')
    OpsRegister()["LSTM"].is_reverse = helper.attr_data(op, 'is_reverse')
    OpsRegister()["LSTM"].use_peepholes = helper.attr_data(op, 'use_peepholes')
    OpsRegister()["LSTM"].num_direction = 1
    OpsRegister()["LSTM"].dropout_param = 1.0
    OpsRegister()["LSTM"].num_layers = 1
    OpsRegister()["LSTM"].input_activation = "null"
    if bool(private_data) is True:
        np_fc_bias = private_data['np_flat_fc_bias']
        np_fc_weight = private_data['np_flat_fc_weight']
        np_fc_outdim = private_data['np_fc_outdim']
        np_lstm_bias = helper.np_param(op, 'Bias')
        np_lstm_weight = helper.np_param(op, 'Weight')
        np_tensors = helper.lstm_fc_tensor_merge_convert(np_fc_outdim, np_lstm_weight, \
            np_lstm_bias, np_fc_weight, np_fc_bias)
        np_weight = np_tensors[0]
        np_bias = np_tensors[1]
        np_weight_shape = map(int, [1] * (4 - len(np_weight.shape)) + list(np_weight.shape))
        np_bias_shape = map(int, [1] * (4 - len(np_bias.shape)) + list(np_bias.shape))
        np_weight_tensor = helper.create_tensor(np_weight.flatten().tolist(), np_weight_shape, FLOAT)
        np_bias_tensor = helper.create_tensor(np_bias.flatten().tolist(), np_bias_shape, FLOAT)
        OpsRegister()["LSTM"].weight_1 = np_weight_tensor
        OpsRegister()["LSTM"].weight_2 = np_bias_tensor
    else:
        OpsRegister()["LSTM"].weight_1 = helper.param_tensor(op, 'Weight')
        OpsRegister()["LSTM"].weight_2 = helper.create_tensor([0], [-1], FLOAT)


############### RNN ###############

@ParserFeedDecorator("Embedding")
def Parser_lookup_table(args):
    op = args[1]
    helper = args[3]
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'W')
    OpsRegister()["Embedding"].weight_1 = weights_tensor
    OpsRegister()["Embedding"].padding_idx = helper.attr_data(op, 'padding_idx')
    OpsRegister()["Embedding"].word_num = weights_shape[2]
    OpsRegister()["Embedding"].emb_dim = weights_shape[3]

@ParserFeedDecorator("SequencePool")
def Parser_sequence_pool(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["SequencePool"].pooltype = helper.attr_data(op, 'pooltype')

@ParserFeedDecorator("Activation")
def Parser_tanh(args):
    OpsRegister()["Activation"].type = "TanH"

@ParserFeedDecorator("SequenceConv")
def Parser_sequence_conv(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter')
    OpsRegister()["SequenceConv"].weight_1 = weights_tensor
    OpsRegister()["SequenceConv"].filter_num = weights_shape[0]
    OpsRegister()["SequenceConv"].kernel_size = weights_shape[-2:]
    OpsRegister()["SequenceConv"].padding_trainable = helper.attr_data(op, 'paddingTrainable')
    OpsRegister()["SequenceConv"].context_stride = helper.attr_data(op, 'contextStride')
    OpsRegister()["SequenceConv"].context_start = helper.attr_data(op, 'contextStart')
    OpsRegister()["SequenceConv"].context_length = helper.attr_data(op, 'contextLength')
    if 'bias' in private_data.keys():
        OpsRegister()["SequenceConv"].bias_term = True
        OpsRegister()["SequenceConv"].weight_2 = private_data['bias']
    else:
        OpsRegister()["SequenceConv"].bias_term = False

@ParserFeedDecorator("CrfDecoding")
def Parser_crf_decoding(args):
    op = args[1]
    helper = args[3]
    [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Transition')
    OpsRegister()["CrfDecoding"].weight_1 = weights_tensor

@ParserFeedDecorator("MatMul")
def Parser_matmul(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    if 'coeff' in private_data.keys():
        coeff = private_data['coeff']
    else:
        coeff = 1.0
    OpsRegister()["MatMul"].transpose_x = helper.attr_data(op, 'transpose_X')
    OpsRegister()["MatMul"].transpose_y = helper.attr_data(op, 'transpose_Y')
    OpsRegister()["MatMul"].coeff = coeff

@ParserFeedDecorator("Scale")
def Parser_scale(args):
    op = args[1]
    helper = args[3]
    scale_val = helper.attr_data(op, 'scale')
    OpsRegister()["Scale"].axis = 0
    OpsRegister()["Scale"].num_axes = 0
    OpsRegister()["Scale"].bias_term = False
    OpsRegister()["Scale"].weight_1 = helper.create_tensor([scale_val], [1, 1, 1, 1], FLOAT)

@ParserFeedDecorator("LayerNorm")
def Parser_layer_norm(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["LayerNorm"].weight_1 = helper.param_tensor(op, 'Scale')
    OpsRegister()["LayerNorm"].weight_2 = helper.param_tensor(op, 'Bias')
    OpsRegister()["LayerNorm"].begin_norm_axis = helper.attr_data(op, 'begin_norm_axis')
    OpsRegister()["LayerNorm"].eps = helper.attr_data(op, 'epsilon')

@ParserFeedDecorator("Scale")
def Parser_dropout(args):
    op = args[1]
    helper = args[3]
    scale_val = 1 - helper.attr_data(op, 'dropout_prob')
    OpsRegister()["Scale"].axis = 0
    OpsRegister()["Scale"].num_axes = 0
    OpsRegister()["Scale"].bias_term = False
    OpsRegister()["Scale"].weight_1 = helper.create_tensor([scale_val], [1, 1, 1, 1], FLOAT)

@ParserFeedDecorator("Scale")
def Parser_elementwise_mul(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    if helper.is_persistable_param(op, 'Y'):
        OpsRegister()["Scale"].weight_1 = helper.param_tensor(op, 'Y')
    else:
        OpsRegister()["Scale"].weight_1 = helper.create_tensor([1], [1, 1, 1, 1], FLOAT) # developing
    OpsRegister()["Scale"].axis = helper.attr_data(op, 'axis')
    OpsRegister()["Scale"].num_axes = 1
    if 'bias' in private_data.keys():
        OpsRegister()["Scale"].bias_term = True
        OpsRegister()["Scale"].weight_2 = private_data['bias']
    else:
        OpsRegister()["Scale"].bias_term = False

@ParserFeedDecorator("Activation")
def Parser_relu6(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Activation"].type = "ClippedRelu"
    OpsRegister()["Activation"].clip_relu_num = helper.attr_data(op, 'threshold')

@ParserFeedDecorator("ReLU")
def Parser_leaky_relu(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["ReLU"].alpha = helper.attr_data(op, 'alpha')

@ParserFeedDecorator("Activation")
def Parser_prelu(args):
    op = args[1]
    helper = args[3]
    mode = helper.attr_data(op, 'mode')
    OpsRegister()["Activation"].type = "PReLU"
    OpsRegister()["Activation"].weight_1 = helper.param_tensor(op, 'Alpha')
    if mode == "all":
        OpsRegister()["Activation"].channel_shared = True
    elif mode == "channel":
        OpsRegister()["Activation"].channel_shared = False
    else:
        raise NameError('ERROR: Unknown Prelu channel_shared param.')

@ParserFeedDecorator("Flatten")
def Parser_flatten(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Flatten"].start_axis = helper.attr_data(op, 'axis')
    OpsRegister()["Flatten"].end_axis = -1

@ParserFeedDecorator("assign_value")
def Parser_assign_value(args):
    pass

@ParserFeedDecorator("shape")
def Parser_shape(args):
    pass

@ParserFeedDecorator("fake_quantize_abs_max")
def Parser_fake_quantize_abs_max(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("fake_dequantize_max_abs")
def Parser_fake_dequantize_max_abs(args):
    """
    A placeholder for an empty function.
    """
    pass

FLUID_NODE_FILLER = {
    "feed":OpsParam().set_parser(Parser_feed),
    "conv2d":OpsParam().set_parser(Parser_conv2d),
    "conv2d_transpose":OpsParam().set_parser(Parser_conv2d_transpose),
    "elementwise_add":OpsParam().set_parser(Parser_sum),
    "relu":OpsParam().set_parser(Parser_relu),
    "pool2d":OpsParam().set_parser(Parser_pool2d),
    "mul":OpsParam().set_parser(Parser_mul),
    "softmax":OpsParam().set_parser(Parser_softmax),
    "sigmoid":OpsParam().set_parser(Parser_sigmoid),
    "axpy":OpsParam().set_parser(Parser_axpy),
    "batch_norm":OpsParam().set_parser(Parser_batch_norm),
    "disc_bn":OpsParam().set_parser(Parser_scale_disc_bn),
    "scale_of_bn":OpsParam().set_parser(Parser_scale_of_bn),
    "elementwise_mul":OpsParam().set_parser(Parser_elementwise_mul),
    "split_ins":OpsParam().set_parser(Parser_split_ins),
    "depthwise_conv2d":OpsParam().set_parser(Parser_conv2d),
    "reshape":OpsParam().set_parser(Parser_reshape),
    "reshape2":OpsParam().set_parser(Parser_reshape),
    "concat":OpsParam().set_parser(Parser_concat),
    "transpose":OpsParam().set_parser(Parser_transpose),
    "transpose2":OpsParam().set_parser(Parser_transpose),
    "prior_box":OpsParam().set_parser(Parser_prior_box),
    "box_coder":OpsParam().set_parser(Parser_box_coder),
    "multiclass_nms":OpsParam().set_parser(Parser_multiclass_nms),
    "concat_btw_priorbox_boxcoder":OpsParam().set_parser(Parser_concat_btw_priorbox_boxcoder),
    "im2sequence":OpsParam().set_parser(Parser_im2sequence),
    "gru":OpsParam().set_parser(Parser_gru),
    "sum":OpsParam().set_parser(Parser_sum),
    "lrn":OpsParam().set_parser(Parser_lrn),
    "top_k":OpsParam().set_parser(Parser_top_k),
    "ctc_align":OpsParam().set_parser(Parser_ctc_align),
    "cast":OpsParam().set_parser(Parser_cast),
    "lookup_table":OpsParam().set_parser(Parser_lookup_table),
    "sequence_pool":OpsParam().set_parser(Parser_sequence_pool),
    "tanh":OpsParam().set_parser(Parser_tanh),
    "sequence_conv":OpsParam().set_parser(Parser_sequence_conv),
    "crf_decoding":OpsParam().set_parser(Parser_crf_decoding),
    "lstm":OpsParam().set_parser(Parser_lstm),
    "matmul":OpsParam().set_parser(Parser_matmul),
    "layer_norm":OpsParam().set_parser(Parser_layer_norm),
    "dropout":OpsParam().set_parser(Parser_dropout),
    "scale":OpsParam().set_parser(Parser_scale),
    "flatten":OpsParam().set_parser(Parser_flatten),
    "flatten2":OpsParam().set_parser(Parser_flatten),
    "assign_value":OpsParam().set_parser(Parser_assign_value),
    "shape":OpsParam().set_parser(Parser_shape),
    "relu6":OpsParam().set_parser(Parser_relu6),
    "leaky_relu":OpsParam().set_parser(Parser_leaky_relu),
    "prelu":OpsParam().set_parser(Parser_prelu),
    "split":OpsParam().set_parser(Parser_slice),
    "fake_quantize_abs_max":OpsParam().set_parser(Parser_fake_quantize_abs_max),
    "fake_dequantize_max_abs":OpsParam().set_parser(Parser_fake_dequantize_max_abs),
}
