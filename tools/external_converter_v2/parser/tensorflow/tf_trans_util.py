import tensorflow as tf
import numpy as np
from tensorflow.core.framework import types_pb2, tensor_pb2
from google.protobuf import text_format
from med_graph import MedNodeUtil, MedGraphUtil

TF_TO_ANAKIN_DTYPE = {
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT8: np.int8,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_INT64: np.int64,
    # types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    # types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    # types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    types_pb2.DT_BOOL: np.bool,
}

ANAKIN_VALID_ATTRIBUTES = {
    'p', 'bias', 'axes', 'pads', 'mean', 'activation_beta', 'spatial_scale', 'broadcast', 'pooled_shape', 'high',
    'activation_alpha', 'is_test', 'hidden_size', 'activations', 'beta', 'input_as_shape', 'drop_states', 'alpha',
    'momentum', 'scale', 'axis', 'dilations', 'transB', 'axis_w', 'blocksize', 'output_sequence', 'mode', 'perm',
    'min', 'seed', 'ends', 'paddings', 'to', 'gamma', 'width_scale', 'normalize_variance', 'group', 'ratio',
    'values',
    'dtype', 'output_shape', 'spatial', 'split', 'input_forget', 'keepdims', 'transA', 'auto_pad', 'border', 'low',
    'linear_before_reset', 'height_scale', 'output_padding', 'shape', 'kernel_shape', 'epsilon', 'size', 'starts',
    'direction', 'max', 'clip', 'across_channels', 'value', 'strides', 'extra_shape', 'scales', 'k', 'sample_size',
    'blocksize', 'epsilon', 'momentum'
}


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    assert isinstance(tensor, tensor_pb2.TensorProto)
    is_raw = False
    if tensor.tensor_content:
        data = tensor.tensor_content
        is_raw = True
    elif tensor.float_val:
        data = tensor.float_val
    elif tensor.dcomplex_val:
        data = tensor.dcomplex_val
    elif tensor.int_val:
        data = tensor.int_val
    elif tensor.bool_val:
        data = tensor.bool_val
    elif tensor.dtype == tf.int32:
        data = [0]
    elif tensor.dtype == tf.int64:
        data = [0]
    elif tensor.dtype == tf.float32:
        data = [0.]
    elif tensor.string_val:
        data = tensor.string_val
    else:
        raise ValueError('tensor data not supported')
    return [is_raw, data]


def map_tf_dtype(dtype):
    '''
    wraper for TF_TO_ANAKIN_DTYPE
    :param dtype:
    :return:
    '''
    return TF_TO_ANAKIN_DTYPE.get(dtype)


def get_shape(node):
    """Get shape from tensorflow node."""
    # FIXME: do we use this?
    dims = None
    try:
        if node.type == "Const":
            shape = node.get_attr("value").tensor_shape
            dims = [int(d.size) for d in shape.dim]
        else:
            shape = node.get_attr("shape")
            dims = [d.size for d in shape.dim]
        if shape[0] is not None or shape[0] == -1:
            shape[0] = 1
    except Exception as ex:
        pass
    return dims


def tf_to_anakin_tensor(tensor):
    """Convert tensorflow tensor to anakin med tensor."""
    new_type = TF_TO_ANAKIN_DTYPE[tensor.dtype]
    tdim = tensor.tensor_shape.dim
    dims = [d.size for d in tdim]
    # FIXME: something is fishy here
    if dims == [0]:
        dims = [1]
    is_raw, data = get_tf_tensor_data(tensor)
    dim_size = 1
    for i in dims:
        dim_size *= i
    # print(type(data),data,tensor.dtype,is_raw)
    if is_raw:
        if len(dims) > 0:
            anakin_tensor = np.frombuffer(data, map_tf_dtype(tensor.dtype))
            anakin_tensor = anakin_tensor.reshape(dims)
        else:
            anakin_tensor = np.zeros(0)
        return anakin_tensor
    elif dim_size > 1 and len(data) == 1:

        return np.array([data] * dim_size).reshape(dims)
    else:
        return data


def load_graph(graph_path):
    '''
    load tensorflow graph
    :param graph_path:
    :return:
    '''
    if graph_path.endswith('.pbtxt'):
        input_binary = False
    else:
        input_binary = True

    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    if graph_path.endswith('.pb') or graph_path.endswith('.pbtxt'):
        mode = "rb" if input_binary else "r"
        with tf.gfile.FastGFile(graph_path, mode) as f:
            if input_binary:
                graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), graph_def)
    else:
        tf.train.import_meta_graph(graph_path, clear_devices=True)
    tf.import_graph_def(graph_def, name='graph')
    return graph


def find_layout_in(node, graph):
    if node['ak_type'] in ('Dense'):
        return None
    if 'data_format' in node['tf_attr']:
        return node['tf_attr']['data_format']
    elif len(node['input']) > 0:
        return find_layout_in(graph[node['input'][0]['name']], graph)
    else:
        return None


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def spatial_map(shape, perm):
    '''
    convert shape in different layout
    :param shape:
    :param perm:
    :return:
    '''
    # print('HI ',type(shape),shape)
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def trans_move_attr(attr, format):
    '''
    get h,w shape from attr
    :param attr:
    :param format:
    :return:
    '''
    if attr is None:
        return attr
    if len(attr) == 2:
        return attr
    if format == 'NHWC':
        n, h, w, c = attr
    elif format == 'NCHW':
        n, c, h, w = attr
    else:
        raise Exception('not support format ' + format)
    return [h, w]


def add_special_pad(padding, tf_node, graph):
    '''
    add pad op to solve different pad method in caffe and tensorflow
    :param padding:
    :param tf_node:
    :param graph:
    :return:
    '''
    # print(tf_node['name'])
    assert len(tf_node['input']) <= 2
    now_shape = tf_node['input'][0]['shape']
    tar_shape = now_shape[:]
    tar_shape[1] = now_shape[1] + padding[0] + padding[1]
    tar_shape[2] = now_shape[2] + padding[2] + padding[3]
    tf_name = tf_node['name']
    padding_node = {'name': tf_node['name'] + '_pad', 'ak_type': 'Pad', 'type': None, 'visted': True,
                    'ak_attr': {'pad_c': [0, 0], 'pad_h': [padding[0], padding[1]], 'pad_w': [padding[2], padding[3]]},
                    'input': [tf_node['input'][0]], 'output': [{'name': tf_name, 'shape': tar_shape}]}

    input_0 = graph[tf_node['input'][0]['name']]
    input_0['output'] = MedNodeUtil.replace_name_with_list(input_0['output'], tf_name,
                                                           [{'name': padding_node['name'], 'shape': now_shape}])
    tf_node['input'][0] = {'name': padding_node['name'], 'shape': tar_shape}
    graph[padding_node['name']] = padding_node


def parse_slim_flatten(tf_node, graph):
    '''
    parse shape for tensorflow graph
    :param tf_node:
    :param graph:
    :return:
    '''
    # try:

    assert len(tf_node['output']) == 1
    get_shape_node = graph[tf_node['input'][0]['name']]
    pack_node = graph[tf_node['output'][0]['name']]
    assert get_shape_node['type'] == 'Shape'
    assert pack_node['type'] == 'Pack'
    assert len(pack_node['output']) == 1
    reshape_node = graph[pack_node['output'][0]['name']]
    assert reshape_node['type'] == 'Reshape'
    assert reshape_node['input'][0]['name'] == get_shape_node['input'][0]['name']

    tf_node['visted'] = True
    get_shape_node['visted'] = True
    pack_node['visted'] = True
    reshape_node['visted'] = True

    the_node = MedNodeUtil.new_med_node(name=tf_node['name'] + '_flatten')
    graph[the_node['name']] = the_node

    the_node['type'] = 'Flatten'
    the_node['ak_type'] = 'Flatten'
    the_node['input'] = get_shape_node['input']
    the_node['output'] = reshape_node['output']
    the_node['visted'] = True
    MedNodeUtil.redirecto_outputs_input_to_this_any(
        the_node, graph, reshape_node['name'], the_node['name'], the_node['output'][0]['shape'])
    MedNodeUtil.redirecto_inputs_output_to_this_any(
        the_node, graph, get_shape_node['name'], the_node['name'], the_node['input'][0]['shape'])
    pre_out = graph[the_node['input'][0]['name']]['output']
    for index, out in enumerate(pre_out):
        if out['name'] == reshape_node['name']:
            del pre_out[index]

    # print(the_node['output'])
    # print(graph[the_node['output'][0]['name']]['input'])
    # exit()

    # except Exception,e:
    #     raise e


def parse_Identity(tf_node, graph):
    '''
    remove identity in tensorflow graph
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    outputs = tf_node['output']
    input_0 = tf_node['input'][0]
    in_node = graph[input_0['name']]

    for next in outputs:
        next_name = next['name']
        next_node = graph[next_name]
        next_node['input'] = [input_0 if i['name'] == tf_node['name'] else i for i in next_node['input']]
        in_node['output'] = MedNodeUtil.replace_name_with_list(
            in_node['output'], tf_node['name'], outputs)


def parse_Shape(tf_node, graph):
    '''
    parse shape for tensorflow graph
    :param tf_node:
    :param graph:
    :return:
    '''
    assert len(tf_node['input']) == 1
    tf_node['type'] = 'Const'
    tf_node['tf_attr']['value'] = tf_node['input'][0]['shape']
    tf_node['tf_attr']['dtype'] = np.int32


def parse_Squeeze(tf_node, graph):
    '''
    convert squeeze to reshape
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Reshape'
    tf_node['ak_attr']['shape'] = tf_node['output'][0]['shape']
    tf_node['output'] = [i for i in tf_node['output'] if graph[i['name']]['type'] != 'Const']


def parse_Placeholder(tf_node, graph):
    '''
    conver placeholder to input op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Input'
    if len(tf_node['output'][0]['shape']) == 4:
        tf_node['ak_attr']['shape'] = spatial_map(tf_node['output'][0]['shape'], NHWC_TO_NCHW)
    else:
        tf_node['ak_attr']['shape'] = tf_node['output'][0]['shape']


def parse_Pad(tf_node, graph):
    '''
    convert pad op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Pad'
    pad_shape_node = graph[tf_node['input'][1]['name']]
    assert pad_shape_node['type'] == 'Const'
    pad_shape = pad_shape_node['tf_attr']['value']
    ak_attr = tf_node['ak_attr']
    ak_attr['pad_c'] = pad_shape[3].flatten().tolist()
    ak_attr['pad_h'] = pad_shape[1].flatten().tolist()
    ak_attr['pad_w'] = pad_shape[2].flatten().tolist()


def parse_Transpose(tf_node, graph):
    '''
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Permute'
    assert len(tf_node['input']) == 2
    arg_node = graph[tf_node['input'][1]['name']]
    assert arg_node['type'] == 'Const'
    tf_node['ak_attr']['dims'] = arg_node['tf_attr']['value'].flatten().tolist()
    print(tf_node['ak_attr']['dims'], type(tf_node['ak_attr']['dims']))
    # exit()
    pass


def parse_Softmax(tf_node, graph):
    '''
    convert softmax op, default axis is 3
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Softmax'
    tf_node['ak_attr']['axis'] = 3


def parse_Reshape(tf_node, graph):
    '''
    convert reshape op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Reshape'
    shape = graph[tf_node['input'][1]['name']]
    assert shape['type'] == 'Const'
    assert shape['tf_attr']['dtype'] == np.int32
    tf_node['ak_attr']['shape'] = shape['tf_attr']['value']
    tf_node['input'] = tf_node['input'][:1]
    MedNodeUtil.retain_input(tf_node, [tf_node['input'][0]])


def parse_Act(tf_node, graph):
    '''
    convert activate op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Activation'
    if tf_node['type'] == 'Relu':
        tf_node['ak_type'] = 'ReLU'
        tf_node['ak_attr']['type'] = 'Relu'
    elif tf_node['type'] == 'Relu6':
        tf_node['ak_type'] = 'Activation'
        tf_node['ak_attr']['type'] = 'ClippedRelu'
        tf_node['ak_attr']['clip_relu_num'] = 6
    else:
        raise Exception('un handel activation ' + str(tf_node['type']))


def parse_Concat(tf_node, graph):
    '''
    convert concat op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Concat'
    N = tf_node['tf_attr']['N']
    axis_node = graph[tf_node['input'][N]['name']]
    assert axis_node['type'] == 'Const'
    nhwc2hchw = {0: 0, 1: 2, 2: 3, 3: 1}
    tf_node['ak_attr']['axis'] = nhwc2hchw[axis_node['tf_attr']['value'][0]]


def parse_Add(tf_node, graph):
    '''
    convert add op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    print(tf_node)
    assert len(tf_node['input']) == 2
    input_0 = graph[tf_node['input'][0]['name']]
    input_1 = graph[tf_node['input'][1]['name']]
    ak_attr = tf_node['ak_attr']

    if input_0['type'] == 'Const' and input_1['type'] == 'Const':
        assert False, 'have not handle'
    elif input_0['type'] == 'Const' and input_1['type'] != 'Const':
        assert False, 'not support'
    elif input_0['type'] != 'Const' and input_1['type'] == 'Const':
        assert False, 'not support'
    else:
        tf_node['ak_type'] = 'Eltwise'
        ak_attr['type'] = 'Add'


def parse_Mean(tf_node, graph):
    '''
    convert mean op to pooling op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Pooling'
    ak_attr = tf_node['ak_attr']
    tf_attr = tf_node['tf_attr']
    ak_attr['type'] = 'AVG'
    reduction_shape = None
    keep_dims = tf_attr['keep_dims']

    if len(tf_node['input']) > 1:
        reduction_shape_node = graph[tf_node['input'][1]['name']]
        assert reduction_shape_node['type'] == 'Const'
        reduction_shape = reduction_shape_node['tf_attr']['value'].flatten().tolist()
    assert reduction_shape is not None
    assert keep_dims is True
    # print('reduction ',reduction_shape,tf_node['name'])
    # assert reduction_shape == [1, 2]
    ak_attr['strides'] = [1, 1]
    ak_attr['window'] = [tf_node['input'][0]['shape'][reduction_shape[0]],
                         tf_node['input'][0]['shape'][reduction_shape[1]]]
    ak_attr['padding'] = [0, 0]


def parse_Pooling(tf_node, graph):
    '''
    convert pooling op
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Pooling'
    ak_attr = tf_node['ak_attr']
    tf_attr = tf_node['tf_attr']
    # TODO TF default pad is exclusive method
    map_tf_pool = {'MaxPool': 'MAX',
                   'AvgPool': 'AVGEXC'}

    ak_attr['type'] = map_tf_pool[tf_node['type']]

    strides = tf_attr['strides']
    ksizes = tf_attr['ksize']
    padding = tf_attr['padding']
    dilations = tf_attr.get('dilations')
    data_format = tf_attr['data_format'].decode()

    kernel_shape = trans_move_attr(ksizes, data_format)
    strides = trans_move_attr(strides, data_format)

    padding = cal_padding(padding, kernel_shape, strides, dilations, data_format,
                          tf_node['input'][0]['shape'], tf_node['output'][0]['shape'])

    if padding[0] != padding[1] or padding[2] != padding[3]:
        add_special_pad(padding, tf_node, graph)
        padding = [0, 0]
    else:
        padding = [padding[0], padding[2]]

    ak_attr['window'] = kernel_shape
    ak_attr['padding'] = padding
    ak_attr['strides'] = strides
    ak_attr['cmp_out_shape_floor_as_conv'] = True


def cal_padding(padding, kernel_shape, strides, dilations, data_format, input_shape, output_shape, spatial=2):
    '''
    convert pad string to pad list
    :param padding:
    :param kernel_shape:
    :param strides:
    :param dilations:
    :param data_format:
    :param input_shape:
    :param output_shape:
    :param spatial:
    :return:
    '''
    pads = [0] * spatial * 2
    if type(padding) == bytes:
        padding = padding.decode()
        if dilations is None:
            dilations = [1] * spatial * 2
        if padding == 'SAME':
            pads = [0] * spatial * 2
            if data_format == 'NHWC':
                # print('in out shape = ',input_shape,output_shape)
                input_shape = spatial_map(input_shape, NHWC_TO_NCHW)
                output_shape = spatial_map(output_shape, NHWC_TO_NCHW)
            elif data_format != 'NCHW':
                raise Exception('not suppor format ' + data_format)
            for i in range(spatial):
                pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                pad = max(pad, 0)
                pads1 = pad // 2
                pads2 = pad - pad // 2
                pads[i * 2 + 0] = pads1
                pads[i * 2 + 1] = pads2
            return pads
        elif padding == 'VALID':
            return pads
        else:
            raise ValueError("invalid padding value: " + padding)


def get_const_from_biinput(inputs, graph):
    '''
    search const input of node`s input
    :param inputs:
    :param graph:
    :return:
    '''
    assert len(inputs) <= 2
    input_0 = graph[inputs[0]['name']]
    input_1 = graph[inputs[1]['name']]
    if input_0['type'] == 'Const':
        return input_0['tf_attr']['value']
    elif len(inputs) == 2 and input_1['type'] == 'Const':
        return input_1['tf_attr']['value']
    return None


def get_bias(tf_node, graph):
    '''
    try to fetch const value form node
    :param tf_node:
    :param graph:
    :return:
    '''
    bias_weight = None
    output = tf_node['output']
    output_0 = graph[output[0]['name']]
    if len(output) == 1 and (output_0['type'] == 'Add' or output_0['type'] == 'BiasAdd'):
        bias_weight = get_const_from_biinput(output_0['input'], graph)

        if bias_weight is not None:
            output_0['visted'] = True
            tf_node['output'] = output_0['output']
            MedNodeUtil.redirecto_outputs_input_to_this(output_0, graph, tf_node['name'], output[0]['shape'])

    return bias_weight


def fix_Dense(tf_node, graph):
    input_node = graph[tf_node['input'][0]['name']]
    layout = find_layout_in(input_node, graph)
    print(tf_node['name'], tf_node['input'], layout, type(layout))
    if layout == 'NHWC':
        if input_node['ak_type'] in ('Flatten'):
            input_node = graph[input_node['input'][0]['name']]
            shape = input_node['output'][0]['shape']
            weights = tf_node['ak_attr']['weights']
            full_shape = [i for i in shape if i is not None]
            full_shape.append(weights.shape[1])
            weights = weights.reshape(full_shape)
            weights = weights.transpose((2, 0, 1, 3))
            tf_node['ak_attr']['weights'] = weights.reshape(tf_node['ak_attr']['weights'].shape)


def parse_Conv2D(tf_node, graph):
    '''
    convert conv2D to convolution
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Convolution'
    weights_node = graph[tf_node['input'][1]['name']]

    assert weights_node['type'] == 'Const'
    weights = weights_node['tf_attr']['value']
    weights = weights.transpose((3, 2, 0, 1))

    tf_attr = tf_node['tf_attr']
    data_format = tf_attr['data_format'].decode()

    padding = tf_attr['padding']
    dilations = trans_move_attr(tf_attr['dilations'], data_format)
    strides = trans_move_attr(tf_attr['strides'], data_format)
    kernel_shape = weights_node['output'][0]['shape'][:2]
    # print('name ',tf_node['name'],input_node['name'],input_node['out_shape'])
    padding = cal_padding(padding, kernel_shape, strides, dilations, data_format, tf_node['input'][0]['shape'],
                          tf_node['output'][0]['shape'])
    if padding[0] != padding[1] or padding[2] != padding[3]:
        add_special_pad(padding, tf_node, graph)
        padding = [0, 0]
    else:
        padding = [padding[0], padding[2]]

    group = 1
    if tf_node['type'] == 'DepthwiseConv2dNative':
        weights = weights.transpose((1, 0, 2, 3))
        group = weights.shape[0]
        out_c = weights.shape[0] * weights.shape[1]
        weights = weights.reshape(out_c, 1, weights.shape[2], weights.shape[3])

    ak_attr = tf_node['ak_attr']
    ak_attr['weights'] = weights
    ak_attr['padding'] = padding
    ak_attr['dilations'] = dilations
    ak_attr['strides'] = strides
    ak_attr['bias_weights'] = get_bias(tf_node, graph)
    ak_attr['group'] = group
    MedNodeUtil.retain_input(tf_node, [tf_node['input'][0]])


def parse_MatMul(tf_node, graph):
    '''
    convert matmul to dense
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Dense'
    in_name_0 = tf_node['input'][0]['name']
    in_name_1 = tf_node['input'][1]['name']
    in_type_0 = graph[in_name_0]['type']
    in_type_1 = graph[in_name_1]['type']
    if in_type_0 == 'Const' and in_type_1 == 'Const':
        raise Exception('Whate hannpend both const')
    elif in_type_1 == 'Const' and tf_node['tf_attr']['transpose_a'] != True:
        weights = graph[in_name_1]['tf_attr']['value']
        tf_node['ak_attr']['trans_weights'] = not tf_node['tf_attr']['transpose_b']
        tf_node['ak_attr']['weights'] = weights
        MedNodeUtil.retain_input(tf_node, [tf_node['input'][0]])
    elif in_type_0 == 'Const' and tf_node['tf_attr']['transpose_b'] != True:
        weights = graph[in_name_1]['tf_attr']['value']
        tf_node['ak_attr']['trans_weights'] = tf_node['tf_attr']['transpose_a']
        tf_node['ak_attr']['weights'] = weights
        MedNodeUtil.retain_input(tf_node, [tf_node['input'][1]])
    else:
        raise Exception('can`t parse ' + str(tf_node))
    tf_node['ak_attr']['bias_weights'] = get_bias(tf_node, graph)


def parse_fusionReshape(tf_node, graph):
    '''
    convert reshape
    :param tf_node:
    :param graph:
    :return:
    '''
    assert tf_node['type'] == 'Reshape'
    input_0 = graph[tf_node['input'][0]['name']]
    input_1 = graph[tf_node['input'][1]['name']]

    if not (input_0['type'] == 'Const' and input_1['type'] == 'Const'): return

    tf_node['type'] = 'Const'
    const_value = input_0['tf_attr']['value']
    shape = input_1['tf_attr']['value']
    new_const_value = const_value.reshape(shape)
    tf_node['tf_attr'] = {'value': new_const_value}
    tf_node['input'] = []


def parse_CustmerBatchNorm(tf_node, graph):
    '''convert user custmer batchnorm to scale'''
    assert tf_node['type'] == 'Rsqrt'
    assert len(tf_node['input']) == 1 and len(tf_node['output']) == 1
    rsqrt_mul_node = graph[tf_node['output'][0]['name']]
    add_rsqrt_node = graph[tf_node['input'][0]['name']]
    assert len(add_rsqrt_node['input']) == 2 and len(add_rsqrt_node['output']) == 1
    eps_node = graph[add_rsqrt_node['input'][1]['name']]
    var_node = graph[add_rsqrt_node['input'][0]['name']]

    assert eps_node['type'] == 'Const' and var_node['type'] == 'Const'
    epse_value = eps_node['tf_attr']['value']
    var_value = var_node['tf_attr']['value'].flatten()
    assert rsqrt_mul_node['type'] == 'Mul' and len(rsqrt_mul_node['input']) == 2 and len(rsqrt_mul_node['output']) == 2
    const_mul_node_input_1 = graph[rsqrt_mul_node['input'][1]['name']]
    if MedGraphUtil.check_one_of_input_is_const(graph[rsqrt_mul_node['output'][0]['name']], graph):
        mul_mul_1_node = graph[rsqrt_mul_node['output'][1]['name']]
        mul_mul_2_node = graph[rsqrt_mul_node['output'][0]['name']]
    else:
        mul_mul_1_node = graph[rsqrt_mul_node['output'][0]['name']]
        mul_mul_2_node = graph[rsqrt_mul_node['output'][1]['name']]

    assert mul_mul_1_node['type'] == 'Mul' and mul_mul_2_node['type'] == 'Mul' and const_mul_node_input_1[
        'type'] == 'Const'
    assert len(mul_mul_1_node['output']) == 1 and len(mul_mul_2_node['output']) == 1
    alpha_value = const_mul_node_input_1['tf_attr']['value'].flatten()
    mul_add_node = graph[mul_mul_1_node['output'][0]['name']]
    mul2_sub_node = graph[mul_mul_2_node['output'][0]['name']]
    mean_node = graph[mul_mul_2_node['input'][0]['name']]

    assert mul_add_node['type'] == 'Add' and mul2_sub_node['type'] == 'Sub' and mean_node['type'] == 'Const'
    beta_node = graph[mul2_sub_node['input'][0]['name']]

    assert beta_node['type'] == 'Const'
    beta_value = beta_node['tf_attr']['value'].flatten()
    mean_value = mean_node['tf_attr']['value'].flatten()
    assert mul_add_node['input'][1]['name'] == mul2_sub_node['name']

    mul_mul_1_node['visted'] = True
    mul_mul_2_node['visted'] = True
    rsqrt_mul_node['visted'] = True
    mul_add_node['visted'] = True
    add_rsqrt_node['visted'] = True
    mul2_sub_node['visted'] = True
    tf_node['visted'] = True

    var = np.sqrt(var_value + epse_value)
    np_scale = alpha_value / var
    np_bias = beta_value - np_scale * mean_value
    mul_mul_1_node['output'] = mul_add_node['output']
    mul_mul_1_node['type'] = 'CustomerBN'
    mul_mul_1_node['ak_type'] = 'Scale'

    MedNodeUtil.retain_input(mul_mul_1_node, [mul_mul_1_node['input'][0]])
    MedNodeUtil.redirecto_outputs_input_to_this(mul_add_node, graph, mul_mul_1_node['name'],
                                                mul_mul_1_node['output'][0]['shape'])
    ak_attr = mul_mul_1_node['ak_attr']
    ak_attr['scale_weights'] = np_scale.astype('float32')
    ak_attr['bias_weights'] = np_bias.astype('float32')
    pass


def parse_BatchNorm(tf_node, graph):
    '''
    convert fused batchnorm to scale
    :param tf_node:
    :param graph:
    :return:
    '''
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Scale'
    assert len(tf_node['input']) == 5
    alpha_node = graph[tf_node['input'][1]['name']]
    beta_node = graph[tf_node['input'][2]['name']]
    mean_node = graph[tf_node['input'][3]['name']]
    var_node = graph[tf_node['input'][4]['name']]
    assert beta_node['type'] == 'Const' and mean_node['type'] == 'Const' and var_node['type'] == 'Const' and alpha_node[
        'type'] == 'Const'
    tf_attr = tf_node['tf_attr']
    eps = tf_attr['epsilon']
    var_value = var_node['tf_attr']['value'].flatten()

    alpha_value = alpha_node['tf_attr']['value'].flatten()
    mean_value = mean_node['tf_attr']['value'].flatten()
    beta_value = beta_node['tf_attr']['value'].flatten()

    var = np.sqrt(var_value + eps)
    np_scale = alpha_value / var
    np_bias = beta_value - (alpha_value * mean_value / var)
    ak_attr = tf_node['ak_attr']
    ak_attr['scale_weights'] = np_scale.astype('float32')
    ak_attr['bias_weights'] = np_bias.astype('float32')

    MedNodeUtil.retain_input(tf_node, [tf_node['input'][0]])
