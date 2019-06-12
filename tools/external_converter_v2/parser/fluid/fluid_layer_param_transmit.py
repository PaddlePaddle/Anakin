from ..operations import OpsParam, OpsRegister
from ..logger import *
from ..proto import *
from ..proto import helper
from fluid_helper import *
import numpy as np


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
    pass


@ParserFeedDecorator("Input")
def Parser_feed(args):
    op = args[1]
    private_data = args[4]

    layout_dict = {
        2: "NC",
        3: "NHW",
        4: "NCHW",
    }
    private_data = args[4]
    # content_dnn hard decode
    if private_data['net_type'] == 'content_dnn':
        input_shape = [10240, 1, 1, 1]
        OpsRegister()["Input"].max_len = 256
        OpsRegister()["Input"].max_batch = 40
    else:
        input_shape = private_data['input_shape']

    alias = private_data['alias']
    OpsRegister()["Input"].input_shape = input_shape
    OpsRegister()["Input"].alias = alias
    OpsRegister()["Input"].layout = layout_dict[len(input_shape)]


@ParserFeedDecorator("Convolution")
def Parser_conv2d(args):
    node = args[0]
    op = args[1]
    helper = args[3]
    private_data = args[4]
    weights_tensor = None
    weights_shape = None

    if 'scale_1' in private_data:
        node.set_bit_type(INT8)
        [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter', "int8")
        weights_tensor.set_scale(private_data['scale_1'], 'float')
    else:
        node.set_bit_type(FLOAT)
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
    node = args[0]
    op = args[1]
    helper = args[3]
    private_data = args[4]
    weights_tensor = None
    weights_shape = None
    if 'scale_1' in private_data:
        node.set_bit_type(INT8)
        [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter', "int8")
        weights_tensor.set_scale(private_data['scale_1'], 'float')
    else:
        node.set_bit_type(FLOAT)
        [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Filter')
    ## trans weights shape layout   
    ## fluid deconv weights shape layout is (chin, chout/group, h, w)
    ## anakin deconv weights shape layout is (chout/group, chin, h, w)
    tmp = weights_shape[0]
    weights_shape[0] = weights_shape[1]
    weights_shape[1] = tmp
    weights_tensor.set_shape(weights_shape)
    OpsRegister()["Deconvolution"].weight_1 = weights_tensor
    OpsRegister()["Deconvolution"].group = helper.attr_data(op, 'groups')
    OpsRegister()["Deconvolution"].filter_num = weights_shape[0] * OpsRegister()["Deconvolution"].group
    OpsRegister()["Deconvolution"].kernel_size = weights_shape[-2:]
    OpsRegister()["Deconvolution"].strides = helper.attr_data(op, 'strides')
    OpsRegister()["Deconvolution"].padding = helper.attr_data(op, 'paddings')
    OpsRegister()["Deconvolution"].dilation_rate = helper.attr_data(op, 'dilations')
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
        if helper.attr_data(op, 'exclusive', True) is True:
            OpsRegister()["Pooling"].method = 'AVGEXC'
        else:
            OpsRegister()["Pooling"].method = "AVG"
    if helper.attr_data(op, 'ceil_mode') == False:
        OpsRegister()["Pooling"].cmp_out_shape_floor_as_conv = True
    else:
        OpsRegister()["Pooling"].cmp_out_shape_floor_as_conv = False


@ParserFeedDecorator("Dense")
def Parser_mul(args):
    node = args[0]
    op = args[1]
    helper = args[3]
    private_data = args[4]
    weights_needs_trans = True
    weights_tensor = None
    weights_shape = None
    if 'scale_1' in private_data:
        node.set_bit_type(INT8)
        [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Y', "int8", weights_needs_trans)
        weights_tensor.set_scale(private_data['scale_1'], 'float')
    else:
        node.set_bit_type(FLOAT)
        [weights_tensor, weights_shape] = helper.param_tensor_sh(op, 'Y', None, weights_needs_trans)
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
    sections = list(helper.attr_data(op, 'sections'))
    slice_point = list()
    for i in range(len(sections) - 1):
        slice_point.append(sum(sections[:i + 1]))
    OpsRegister()["Slice"].slice_point = slice_point
    OpsRegister()["Slice"].num = helper.attr_data(op, 'num')
    OpsRegister()["Slice"].axis = helper.attr_data(op, 'axis')

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
    elif len(shape) == 2:
        layout = 'NW'
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
    dims = 0
    if fluid_dims < 4:
        n = 4 - len(fluid_dims)
        dims = range(0, n)
        tail_dims = [i + n for i in fluid_dims]
        dims.extend(tail_dims)
    else:
        dims = fluid_dims
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

@ParserFeedDecorator("PriorBox")
def Parser_density_prior_box(args):
    op = args[1]
    helper = args[3]

    OpsRegister()["PriorBox"].fixed_size = helper.attr_data(op, 'fixed_sizes')
    OpsRegister()["PriorBox"].fixed_ratio = helper.attr_data(op, 'fixed_ratios')
    OpsRegister()["PriorBox"].density = map(float, helper.attr_data(op, 'densities'))
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
    op = args[1]
    helper = args[3]
    axis = helper.attr_data(op, 'axis')
    box_normalized = helper.attr_data(op, 'box_normalized')
    variance = helper.attr_data(op, 'variance')

    OpsRegister()["box_coder"].axis = axis
    OpsRegister()["box_coder"].box_normalized = box_normalized
    if type(variance) is int:
        OpsRegister()["box_coder"].variance = helper.create_tensor([variance,], [1, 1, 1, 1,], FLOAT)
    else:
        OpsRegister()["box_coder"].variance = helper.create_tensor(variance, [1, len(variance), 1, 1,], FLOAT)

@ParserFeedDecorator("DetectionOutput")
def Parser_multiclass_nms(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]

    OpsRegister()["DetectionOutput"].share_location = True if private_data['net_type'] == 'SSD' else False
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

word_embedding_tensor = None
word_embedding_owner_op_name = None

@ParserFeedDecorator("Embedding")
def Parser_lookup_table(args):
    op = args[1]
    helper = args[3]

    global word_embedding_tensor
    global word_embedding_owner_op_name

    def _NameNodeMid(op):
        first_outparam = op.output_names[0]
        arg_name = str(op.output(first_outparam)[0]).split('.')[0]
        new_name = op.type + '#' + bytes(op.idx) + '(' + arg_name + ')'
        return new_name

    if word_embedding_tensor is None:
        [weights_tensor, _] = helper.param_tensor_sh(op, 'W')
        word_embedding_owner_op_name = str(_NameNodeMid(op))
        word_embedding_tensor = weights_tensor
    else:
        weights_tensor = TensorProtoIO()
        weights_tensor.set_shared(True)
        weights_tensor.set_shared_from(word_embedding_owner_op_name)

    weights_shape = word_embedding_tensor.get_shape()

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

    Y = helper.var_by_param(op, 'Y')
    if Y.persistable:
        OpsRegister()["Scale"].weight_1 = helper.param_tensor(op, 'Y')
    elif 'fill_constant' in private_data and Y.name in private_data['fill_constant']:
        fill_constant_op = private_data['fill_constant'][Y.name]
        OpsRegister()["Scale"].weight_1 = helper.fill_tensor(fill_constant_op, Y)
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
    OpsRegister()["Activation"].clip_relu_num = float(helper.attr_data(op, 'threshold'))

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

@ParserFeedDecorator("PixelShuffle")
def Parser_pixel_shuffle(args):
    private_data = args[4]
    OpsRegister()["PixelShuffle"].upscale_factor = private_data['factor']


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


@ParserFeedDecorator("fake_dequantize_range_max_abs")
def Parser_fake_dequantize_range_max_abs(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("fake_quantize_range_abs_max")
def Parser_fake_quantize_range_abs_max(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("dequantize")
def Parser_dequantize(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("quantize")
def Parser_quantize(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("increment")
def Parser_increment(args):
    """
    A placeholder for an empty function.
    """
    pass

@ParserFeedDecorator("ShuffleChannel")
def Parser_shuffle_channel(args):
    private_data = args[4]
    OpsRegister()["ShuffleChannel"].group = private_data['group']


@ParserFeedDecorator("Scale")
def Parser_affine_channel(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Scale"].bias_term = True
    OpsRegister()["Scale"].weight_1 = helper.param_tensor(op, 'Scale')
    OpsRegister()["Scale"].weight_2 = helper.param_tensor(op, 'Bias')


@ParserFeedDecorator("RoiAlign")
def Parser_roi_align(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["RoiAlign"].spatial_scale = helper.attr_data(op, 'spatial_scale')
    OpsRegister()["RoiAlign"].pooled_height = helper.attr_data(op, 'pooled_height')
    OpsRegister()["RoiAlign"].pooled_width = helper.attr_data(op, 'pooled_width')
    OpsRegister()["RoiAlign"].sampling_ratio = helper.attr_data(op, 'sampling_ratio')

@ParserFeedDecorator("AnchorGenerator")
def Parser_anchor_generator(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["AnchorGenerator"].anchor_sizes = helper.attr_data(op, 'anchor_sizes')
    OpsRegister()["AnchorGenerator"].aspect_ratios = helper.attr_data(op, 'aspect_ratios')
    OpsRegister()["AnchorGenerator"].variances = helper.attr_data(op, 'variances')
    OpsRegister()["AnchorGenerator"].stride = helper.attr_data(op, 'stride')
    OpsRegister()["AnchorGenerator"].offset = helper.attr_data(op, 'offset')

@ParserFeedDecorator("GenerateProposals")
def Parser_generate_proposals(args):
    op = args[1]
    helper = args[3]

    OpsRegister()["GenerateProposals"].pre_nms_top_n = helper.attr_data(op, 'pre_nms_topN')
    OpsRegister()["GenerateProposals"].post_nms_top_n = helper.attr_data(op, 'post_nms_topN')
    OpsRegister()["GenerateProposals"].nms_thresh = helper.attr_data(op, 'nms_thresh')
    OpsRegister()["GenerateProposals"].min_size = helper.attr_data(op, 'min_size')
    OpsRegister()["GenerateProposals"].eta = helper.attr_data(op, 'eta')

@ParserFeedDecorator("Normalize")
def Parser_norm(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["Normalize"].is_across_spatial = False
    OpsRegister()["Normalize"].is_shared_channel = False
    OpsRegister()["Normalize"].eps = helper.attr_data(op, 'epsilon')
    OpsRegister()["Normalize"].p = 2


@ParserFeedDecorator("Resize")
def Parser_bilinear_interp(args):
    op = args[1]
    helper = args[3]

    scale = helper.attr_data(op, 'scale', None)
    out_w = helper.attr_data(op, 'out_w', -1)
    out_h = helper.attr_data(op, 'out_h', -1)

    if scale is not None:
        OpsRegister()["Resize"].width_scale = scale
        OpsRegister()["Resize"].height_scale = scale
        OpsRegister()["Resize"].out_width = -1
        OpsRegister()["Resize"].out_height = -1
    else:
        OpsRegister()["Resize"].out_width = out_w
        OpsRegister()["Resize"].out_height = out_h

    OpsRegister()["Resize"].method = "BILINEAR_ALIGN"


@ParserFeedDecorator("SequencePoolConcat")
def Parser_seqpool_concat(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]
    OpsRegister()["SequencePoolConcat"].pooltype = helper.attr_data(op, 'pooltype')
    OpsRegister()["SequencePoolConcat"].axis = private_data['axis']
    OpsRegister()["SequencePoolConcat"].slot_num = private_data['slot_num']

@ParserFeedDecorator("Scale")
def Parser_data_norm(args):
    op = args[1]
    helper = args[3]
    batch_size = helper.np_param(op, 'BatchSize')
    batch_square_sum = helper.np_param(op, 'BatchSquareSum')
    batch_sum = helper.np_param(op, 'BatchSum')
    np_means = batch_sum / batch_size
    np_scales = np.sqrt(batch_size / batch_square_sum)
    np_bias = - (np_scales * np_means)
    np_scale_shape = map(int, [1] * (4 - len(np_scales.shape)) + list(np_scales.shape))
    np_bias_shape = map(int, [1] * (4 - len(np_bias.shape)) + list(np_bias.shape))
    np_weight_tensor = helper.create_tensor(np_scales.flatten().tolist(), np_scale_shape, FLOAT)
    np_bias_tensor = helper.create_tensor(np_bias.flatten().tolist(), np_bias_shape, FLOAT)
    OpsRegister()["Scale"].axis = 1
    OpsRegister()["Scale"].num_axes = 1
    OpsRegister()["Scale"].bias_term = True
    OpsRegister()["Scale"].weight_1 = np_weight_tensor
    OpsRegister()["Scale"].weight_2 = np_bias_tensor


@ParserFeedDecorator("fusion_dropout_add_ln_quant")
def Parser_fusion_dropout_add_ln_quant(args):
    pass

@ParserFeedDecorator("dequantize_max_abs_rowwise")
def Parser_dequantize_max_abs_rowwise(args):
    pass

@ParserFeedDecorator("quantize_abs_max_rowwise")
def Parser_quantize_abs_max_rowwise(args):
    pass

@ParserFeedDecorator("fusion_add_relu_dropout_quant")
def Parser_fusion_add_relu_dropout_quant(args):
    pass

@ParserFeedDecorator("fill_constant")
def Parser_fill_constant(args):
    pass

@ParserFeedDecorator("less_than")
def Parser_less_than(args):
    pass

@ParserFeedDecorator("write_to_array")
def Parser_write_to_array(args):
    pass

@ParserFeedDecorator("fill_constant_batch_size_like")
def Parser_fill_constant_batch_size_like(args):
    pass

@ParserFeedDecorator("assign")
def Parser_assign(args):
    op = args[1]
    helper = args[3]

@ParserFeedDecorator("while")
def Parser_while(args):
    pass

@ParserFeedDecorator("beam_search_decode")
def Parser_beam_search_decode(args):
    pass


@ParserFeedDecorator("Resize")
def Parser_nearest_interp(args):
    #pass
    op = args[1]
    helper = args[3]

    out_h = helper.attr_data(op, 'out_h')
    out_w = helper.attr_data(op, 'out_w')
    interp_method = helper.attr_data(op, 'interp_method')
    align_corners = helper.attr_data(op, 'align_corners', False)
    align_mode = helper.attr_data(op, 'align_mode', 0)

    if interp_method == 'nearest':
        if align_corners:
            OpsRegister()["Resize"].method = 'BILINEAR_ALIGN'
        else:
            OpsRegister()["Resize"].method = 'BILINEAR_NO_ALIGN'
        OpsRegister()["Resize"].out_height = out_h
        OpsRegister()["Resize"].out_width = out_w
    else:
        raise Exception('unexpected interp_method={}'.format(interp_method))

@ParserFeedDecorator("yolo_box")
def Parser_yolo_box(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["yolo_box"].class_num = helper.attr_data(op, 'class_num')
    OpsRegister()["yolo_box"].anchors = list(helper.attr_data(op, 'anchors'))
    OpsRegister()["yolo_box"].downsample_ratio = helper.attr_data(op, 'downsample_ratio')
    OpsRegister()["yolo_box"].conf_thresh = helper.attr_data(op, 'conf_thresh')


@ParserFeedDecorator("slice_v2")
def Parser_slice2(args):
    op = args[1]
    helper = args[3]
    OpsRegister()["slice_v2"].ends = list(helper.attr_data(op, 'ends'))
    OpsRegister()["slice_v2"].starts =  list(helper.attr_data(op, 'starts'))
    OpsRegister()["slice_v2"].axes = list(helper.attr_data(op, 'axes'))


@ParserFeedDecorator("reduce")
def Parser_reduce_mean(args):
    op = args[1]
    helper = args[3]
    dim = helper.attr_data(op, 'dim')
    keep_dim = helper.attr_data(op, 'keep_dim')

    OpsRegister()['reduce'].reduce_type = 'Reduce_avg'
    OpsRegister()['reduce'].keep_dim = keep_dim
    if dim is None:
        OpsRegister()['reduce'].reduce_all = True
    elif type(dim) is list:
        OpsRegister()['reduce'].reduce_all = False
        OpsRegister()['reduce'].reduce_dim = dim
    elif type(dim) is int:
        OpsRegister()['reduce'].reduce_all = False
        OpsRegister()['reduce'].reduce_dim = [dim,]
    else:
        raise Exception('unexpected type(dim)={0}'.format(type(dim)))


@ParserFeedDecorator("Argmax")
def Parser_arg_max(args):
    op = args[1]
    helper = args[3]

    OpsRegister()["Argmax"].top_k = 1
    OpsRegister()["Argmax"].axis_term =  True
    OpsRegister()["Argmax"].out_max_value = False
    OpsRegister()["Argmax"].axis = helper.attr_data(op, 'axis')

@ParserFeedDecorator("sequence_expand")
def Parser_sequence_expand(args):
    op = args[1]
    helper = args[3]
    ref_level = helper.attr_data(op, 'ref_level')

    OpsRegister()['sequence_expand'].ref_level = ref_level


@ParserFeedDecorator("Scale")
def Parser_elementwise_div(args):
    op = args[1]
    helper = args[3]
    private_data = args[4]

    axis = helper.attr_data(op, 'axis', -1)
    Y = helper.var_by_param(op, 'Y')
    if Y.persistable:
        weight_1 = helper.param_tensor(op, 'Y')
    elif 'fill_constant' in private_data and Y.name in private_data['fill_constant']:
        fill_constant_op = private_data['fill_constant'][Y.name]
        weight_1 = helper.fill_tensor(fill_constant_op, Y)
    else:
        weight_1 = helper.create_tensor([1], [1, 1, 1, 1], FLOAT)  # developing
    # reverse cache_data
    helper.reverse_cache_data(weight_1.tensor_proto.data)

    OpsRegister()["Scale"].axis = axis
    OpsRegister()["Scale"].num_axes = 1
    OpsRegister()["Scale"].weight_1 = weight_1


@ParserFeedDecorator("box_clip")
def Parser_box_clip(args):
    pass


@ParserFeedDecorator("Reduce")
def Parser_reduce_prod(args):
    op = args[1]
    helper = args[3]
    dim = helper.attr_data(op, 'dim')
    keep_dim = helper.attr_data(op, 'keep_dim')

    OpsRegister()['reduce'].reduce_type = 'Reduce_prod'
    OpsRegister()['reduce'].keep_dim = keep_dim
    if dim is None:
        OpsRegister()['reduce'].reduce_all = True
    elif type(dim) is list:
        OpsRegister()['reduce'].reduce_all = False
        OpsRegister()['reduce'].reduce_dim = dim
    elif type(dim) is int:
        OpsRegister()['reduce'].reduce_all = False
        OpsRegister()['reduce'].reduce_dim = [dim,]
    else:
        raise Exception('unexpected type(dim)={0}'.format(type(dim)))


@ParserFeedDecorator("equal")
def Parser_equal(args):
    pass


@ParserFeedDecorator("split_lod_tensor")
def Parser_split_lod_tensor(args):
    pass


@ParserFeedDecorator("conditional_block")
def Parser_conditional_block(args):
    pass


@ParserFeedDecorator("merge_lod_tensor")
def Parser_merge_lod_tensor(args):
    pass


@ParserFeedDecorator('lod_reset')
def Parser_lod_reset(args):
    """fluid.layers.lod_reset parser
    """
    pass


@ParserFeedDecorator('GroupNormal')
def Parser_group_norm(args):
    """fluid.layers.group_norm parser
    """
    op = args[1]
    helper = args[3]
    private_data = args[4]

    Bias = helper.broad_param_tensor(op, 'Bias', private_data)
    Scale = helper.broad_param_tensor(op, 'Scale', private_data)
    epsilon = helper.attr_data(op, 'epsilon', 0.0)
    groups = helper.attr_data(op, 'groups', 0)

    OpsRegister()['GroupNormal'].has_scale = True
    OpsRegister()['GroupNormal'].scale = Scale
    OpsRegister()['GroupNormal'].has_bias = True
    OpsRegister()['GroupNormal'].bias = Bias
    OpsRegister()['GroupNormal'].eps = epsilon
    OpsRegister()['GroupNormal'].group = groups


@ParserFeedDecorator('fake_quantize_moving_average_abs_max')
def Parser_fake_quantize_moving_average_abs_max(args):
    """fluid.layers.fake_quantize_moving_average_abs_max parser
    """
    pass


@ParserFeedDecorator('Activation')
def Parser_swish(args):
    """fluid.layers.swish parser
    """
    op = args[1]
    helper = args[3]

    beta = helper.attr_data(op, 'beta', 1.0)

    OpsRegister()['Activation'].type = 'Swish'
    OpsRegister()['Activation'].clip_relu_num = beta


@ParserFeedDecorator('reverse_sequence')
def Parser_sequence_reverse(args):
    """paddle.fluid.layers.sequence_reverse parser
    """
    pass


@ParserFeedDecorator('arithmetic')
def Parser_search_seq_arithmetic(args):
    """search_seq_arithmetic parser
    """
    op = args[1]
    helper = args[3]

    op_type = helper.attr_data(op, 'op_type', 0)

    OpsRegister()['arithmetic'].op_type = op_type


@ParserFeedDecorator("Convolution")
def Parser_var_conv_2d(args):
    op = args[1]
    helper = args[3]

    input_channel = helper.attr_data(op, 'InputChannel')
    output_channel = helper.attr_data(op, 'OutputChannel')
    kernel_h = helper.attr_data(op, 'KernelH')
    kernel_w = helper.attr_data(op, 'KernelW')
    stride_h = helper.attr_data(op, 'StrideH')
    stride_w = helper.attr_data(op, 'StrideW')

    [weights_tensor, _] = helper.param_tensor_sh(op, 'W')
    weights_tensor.set_shape([output_channel, input_channel, kernel_h, kernel_w])
    # assert weights_shape == [output_channel, input_channel, kernel_h, kernel_w], \
    #     "weights_shape={0}, output_channel={1}, input_channel={2}, kernel_h={3}, kernel_w={4}".format(
    #         weights_shape, output_channel, input_channel, kernel_h, kernel_w)

    OpsRegister()["Convolution"].weight_1 = weights_tensor
    OpsRegister()["Convolution"].filter_num = output_channel
    OpsRegister()["Convolution"].kernel_size = [kernel_h, kernel_w]
    OpsRegister()["Convolution"].strides = [stride_h, stride_w]
    OpsRegister()["Convolution"].padding = [kernel_h / 2, kernel_w / 2]
    OpsRegister()["Convolution"].dilation_rate = [1, 1]
    OpsRegister()["Convolution"].group = 1
    OpsRegister()["Convolution"].axis = 0
    # TODO: support var_conv_2d has bias condition
    OpsRegister()["Convolution"].bias_term = False
    OpsRegister()["Convolution"].input_channel = input_channel


@ParserFeedDecorator("sequence_depadding")
def Parser_search_seq_depadding(args):
    pass



@ParserFeedDecorator("Gru")
def Parser_search_grnn(args):
    op = args[1]
    helper = args[3]

    [Wi_tensor, _] = helper.param_tensor_sh(op, 'Wi')
    [Wh_tensor, _] = helper.param_tensor_sh(op, 'Wh')
    num_hidden = helper.attr_data(op, 'num_hidden')
    num_input = helper.attr_data(op, 'num_input')

    assert list(Wi_tensor.get_shape()) == [1, 3, num_hidden, num_input], \
        'Wi_tensor.get_shape()={}'.format(Wi_tensor.get_shape())
    assert list(Wh_tensor.get_shape()) == [1, 3, num_hidden, num_hidden], \
        'Wh_tensor.get_shape()={}'.format(Wh_tensor.get_shape())

    # Wi_tensor [1, 3, num_hidden, num_input] => [1, 3, num_input, num_hidden]
    Wi_np_array = np.array(Wi_tensor.get_data())\
        .reshape(Wi_tensor.get_shape())
    Wh_np_array = np.array(Wh_tensor.get_data())\
        .reshape(Wh_tensor.get_shape())

    def gru(weights_wx, weights_wh, word_size, hidden_size):
        weights_wx = weights_wx.flatten().reshape(3, hidden_size, word_size)
        weights_i2h = np.concatenate([weights_wx[0].T, weights_wx[1].T, weights_wx[2].T], axis=1)
        weights_wh = weights_wh.flatten().reshape(3, hidden_size, hidden_size)
        weights_h2h = np.concatenate([weights_wh[1].T, weights_wh[2].T], axis=1)
        weights_h2h = np.concatenate([weights_wh[0].T.flatten(), weights_h2h.flatten()]).reshape(3, hidden_size, hidden_size)
        weights = np.concatenate([weights_i2h.flatten(), weights_h2h.flatten()])
        weights = weights.reshape(1, 1, 1, len(weights))
        return weights

    tensor_tmp2 = gru(Wi_np_array, Wh_np_array, num_input, num_hidden)
    Wi_tensor.set_data(tensor_tmp2.flatten(), 'float')
    Wi_tensor.set_shape(tensor_tmp2.shape)
    tensor_tmp3 = np.zeros([1, 1, 1, 3 * num_hidden])
    Wh_tensor.set_data(tensor_tmp3.flatten(), 'float')
    Wh_tensor.set_shape(tensor_tmp3.shape)
    OpsRegister()["Gru"].weight_1 = Wi_tensor
    OpsRegister()["Gru"].weight_2 = Wh_tensor
    OpsRegister()["Gru"].is_reverse = False
    OpsRegister()["Gru"].gate_activation = "sigmoid"
    OpsRegister()["Gru"].activation = "tanh"
    OpsRegister()["Gru"].gru_formula = "gru_cudnn"


@ParserFeedDecorator('Softmax')
def Parser_search_seq_softmax(args):
    private_data = args[4]

    axis = private_data.get('axis', 1)
    OpsRegister()["Softmax"].axis = axis


@ParserFeedDecorator('aligned_mat_mul')
def Parser_search_aligned_mat_mul(args):
    op = args[1]
    helper = args[3]

    alpha = helper.attr_data(op, 'alpha')
    transpose_X = helper.attr_data(op, 'transpose_X')
    transpose_Y = helper.attr_data(op, 'transpose_Y')

    OpsRegister()["aligned_mat_mul"].coeff = alpha
    OpsRegister()["aligned_mat_mul"].transpose_x = transpose_X
    OpsRegister()["aligned_mat_mul"].transpose_y = transpose_Y


@ParserFeedDecorator('attention_padding_mask')
def Parser_search_attention_padding_mask(args):
    op = args[1]
    helper = args[3]

    pad_id = helper.attr_data(op, 'pad_id')
    mask = helper.attr_data(op, 'mask')

    OpsRegister()['attention_padding_mask'].pad_id = pad_id
    OpsRegister()['attention_padding_mask'].mask = mask


@ParserFeedDecorator('sequence_padding')
def Parser_search_group_padding(args):
    pass


@ParserFeedDecorator('topk_avg_pooling')
def Parser_sequence_topk_avg_pooling(args):
    op = args[1]
    helper = args[3]

    topks = helper.attr_data(op, 'topks')
    channel_num = helper.attr_data(op, 'channel_num')

    OpsRegister()['topk_avg_pooling'].top_ks = topks
    OpsRegister()['topk_avg_pooling'].feat_map_num = channel_num
    OpsRegister()['topk_avg_pooling'].is_pooling_by_row = True


@ParserFeedDecorator('Dense')
def Parser_search_fc(args):
    op = args[1]
    helper = args[3]

    out_size = helper.attr_data(op, 'out_size')
    [weight1_tensor, _] = helper.param_tensor_sh(op, 'W')
    [bias_tensor, _] = helper.param_tensor_sh(op, 'b')

    OpsRegister()['Dense'].weight_1 = weight1_tensor
    if bias_tensor is not None:
        OpsRegister()['Dense'].bias_term = True
        OpsRegister()['Dense'].weight_2 = bias_tensor
    else:
        OpsRegister()['Dense'].bias_term = False
    OpsRegister()["Dense"].out_dim = out_size
    OpsRegister()["Dense"].axis = 1


@ParserFeedDecorator("MatchMatrix")
def Parser_match_matrix_tensor(args):
    op = args[1]
    helper = args[3]

    [weight1_tensor, _] = helper.param_tensor_sh(op, 'W')
    [weight1_n, dim_in, dim_t, dim_in] = weight1_tensor.get_shape()
    assert weight1_n == 1, 'weight1_n={}'.format(weight1_n)
    assert helper.attr_data(op, 'dim_t') == dim_t, \
        'op.dim_t={}'.format(helper.attr_data(op, 'dim_t'))

    OpsRegister()["MatchMatrix"].weight_1 = weight1_tensor
    OpsRegister()["MatchMatrix"].dim_in = dim_in
    OpsRegister()["MatchMatrix"].dim_t = dim_t
    OpsRegister()["MatchMatrix"].linear_term = False
    OpsRegister()["MatchMatrix"].bias_term = False


@ParserFeedDecorator('Dense')
def Parser_search_seq_fc(args):
    op = args[1]
    helper = args[3]

    out_size = helper.attr_data(op, 'out_size')
    has_bias = helper.attr_data(op, 'has_bias', False)
    [weight1_tensor, _] = helper.param_tensor_sh(op, 'W')
    if has_bias:
        [bias_tensor, _] = helper.param_tensor_sh(op, 'b')

    OpsRegister()['Dense'].weight_1 = weight1_tensor
    OpsRegister()['Dense'].bias_term = has_bias
    if has_bias:
        OpsRegister()['Dense'].weight_2 = bias_tensor
    OpsRegister()["Dense"].out_dim = out_size
    OpsRegister()["Dense"].axis = 1


@ParserFeedDecorator('Concat')
def Parser_sequence_concat(args):
    op = args[1]
    helper = args[3]

    OpsRegister()["Concat"].axis = 1


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
    "density_prior_box":OpsParam().set_parser(Parser_density_prior_box),
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
    "quantize":OpsParam().set_parser(Parser_quantize),
    "dequantize":OpsParam().set_parser(Parser_dequantize),
    "fake_quantize_abs_max":OpsParam().set_parser(Parser_fake_quantize_abs_max),
    "fake_quantize_range_abs_max":OpsParam().set_parser(Parser_fake_quantize_range_abs_max),
    "fake_dequantize_max_abs":OpsParam().set_parser(Parser_fake_dequantize_max_abs),
    "fake_dequantize_range_max_abs":OpsParam().set_parser(Parser_fake_dequantize_range_max_abs),
    "pixel_shuffle":OpsParam().set_parser(Parser_pixel_shuffle),
    "shuffle_channel":OpsParam().set_parser(Parser_shuffle_channel),
    # FastRCNN start
    "affine_channel":OpsParam().set_parser(Parser_affine_channel),
    "anchor_generator":OpsParam().set_parser(Parser_anchor_generator),
    "generate_proposals":OpsParam().set_parser(Parser_generate_proposals),
    "roi_align":OpsParam().set_parser(Parser_roi_align),
    # FastRCNN end
    "norm":OpsParam().set_parser(Parser_norm),
    "increment":OpsParam().set_parser(Parser_increment),
    "bilinear_interp":OpsParam().set_parser(Parser_bilinear_interp),
    # feed
    "data_norm":OpsParam().set_parser(Parser_data_norm),
    "seqpool_concat":OpsParam().set_parser(Parser_seqpool_concat),
    # capi
    "fusion_dropout_add_ln_quant":OpsParam().set_parser(Parser_fusion_dropout_add_ln_quant),
    "dequantize_max_abs_rowwise":OpsParam().set_parser(Parser_dequantize_max_abs_rowwise),
    "quantize_abs_max_rowwise":OpsParam().set_parser(Parser_quantize_abs_max_rowwise),
    "fusion_add_relu_dropout_quant":OpsParam().set_parser(Parser_fusion_add_relu_dropout_quant),
    "fill_constant":OpsParam().set_parser(Parser_fill_constant),
    "less_than":OpsParam().set_parser(Parser_less_than),
    "write_to_array":OpsParam().set_parser(Parser_write_to_array),
    "fill_constant_batch_size_like":OpsParam().set_parser(Parser_fill_constant_batch_size_like),
    "assign":OpsParam().set_parser(Parser_assign),
    "while":OpsParam().set_parser(Parser_while),
    "beam_search_decode":OpsParam().set_parser(Parser_beam_search_decode),
    "slice":OpsParam().set_parser(Parser_slice2),
    "nearest_interp":OpsParam().set_parser(Parser_nearest_interp),
    "yolo_box":OpsParam().set_parser(Parser_yolo_box),
    "reduce_mean":OpsParam().set_parser(Parser_reduce_mean),
    "arg_max":OpsParam().set_parser(Parser_arg_max),
    "sequence_expand":OpsParam().set_parser(Parser_sequence_expand),
    "elementwise_div":OpsParam().set_parser(Parser_elementwise_div),
    "box_clip":OpsParam().set_parser(Parser_box_clip),
    "reduce_prod":OpsParam().set_parser(Parser_reduce_prod),
    "equal":OpsParam().set_parser(Parser_equal),
    "split_lod_tensor":OpsParam().set_parser(Parser_split_lod_tensor),
    "conditional_block":OpsParam().set_parser(Parser_conditional_block),
    "merge_lod_tensor": OpsParam().set_parser(Parser_merge_lod_tensor),
    'lod_reset': OpsParam().set_parser(Parser_lod_reset),
    'group_norm': OpsParam().set_parser(Parser_group_norm),
    'fake_quantize_moving_average_abs_max': OpsParam().set_parser(Parser_fake_quantize_moving_average_abs_max),
    'swish': OpsParam().set_parser(Parser_swish),
    'sequence_reverse': OpsParam().set_parser(Parser_sequence_reverse),
    'search_seq_arithmetic': OpsParam().set_parser(Parser_search_seq_arithmetic),
    'search_seq_depadding': OpsParam().set_parser(Parser_search_seq_depadding),
    'search_grnn': OpsParam().set_parser(Parser_search_grnn),
    'search_seq_softmax': OpsParam().set_parser(Parser_search_seq_softmax),
    'search_aligned_mat_mul': OpsParam().set_parser(Parser_search_aligned_mat_mul),
    'search_attention_padding_mask': OpsParam().set_parser(Parser_search_attention_padding_mask),
    'search_group_padding': OpsParam().set_parser(Parser_search_group_padding),
    'sequence_topk_avg_pooling': OpsParam().set_parser(Parser_sequence_topk_avg_pooling),
    'search_fc': OpsParam().set_parser(Parser_search_fc),
    'match_matrix_tensor': OpsParam().set_parser(Parser_match_matrix_tensor),
    'search_seq_fc': OpsParam().set_parser(Parser_search_seq_fc),
    'var_conv_2d': OpsParam().set_parser(Parser_var_conv_2d),
    'sequence_concat': OpsParam().set_parser(Parser_sequence_concat),
}
