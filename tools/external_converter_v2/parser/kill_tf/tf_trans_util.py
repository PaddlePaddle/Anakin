import tensorflow as tf
import numpy as np
from tensorflow.core.framework import types_pb2, tensor_pb2
from google.protobuf import text_format

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

    # print(type(data),data,tensor.dtype,is_raw)
    if is_raw:
        if len(dims) > 0:
            anakin_tensor = np.frombuffer(data, map_tf_dtype(tensor.dtype))
            anakin_tensor = anakin_tensor.reshape(dims)
        else:
            anakin_tensor = np.zeros(0)
        return anakin_tensor
    else:
        return data

def load_graph(graph_path):
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



NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]

def spatial_map(shape, perm):
    # print('HI ',type(shape),shape)
    new_shape = shape[:]
    print(shape)
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape

def trans_move_attr(attr,format):
    if attr is None:
        return attr
    if len(attr)==2:
        return attr
    if format=='NHWC':
        n,h,w,c=attr
    elif format=='NCHW':
        n,c,h,w=attr
    else:
        raise Exception('not support format '+format)
    return [h,w]

def parse_Identity(tf_node,graph):
    tf_node['visted'] = True
    out_name = tf_node['output']
    in_name = tf_node['input'][0]
    in_node=graph[in_name]
    print(out_name)

    for next_name in out_name:
        next_node = graph[next_name]
        next_node['input'] = [in_name if i == tf_node['name'] else i for i in next_node['input']]
        in_node['output']=[next_name if i == tf_node['name'] else i for i in in_node['output']]


def parse_Placeholder(tf_node,graph):
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Input'
    tf_node['ak_attr']['shape']=spatial_map(tf_node['out_shape'],NHWC_TO_NCHW)

def parse_batchnorm(tf_node, graph):
    tf_node['visted'] = True



def parse_Reshape(tf_node,graph):
    tf_node['visted'] = True
    tf_node['ak_type']='Reshape'
    shape=graph[tf_node['input'][1]]
    assert shape['type']=='Const'
    assert shape['tf_attr']['dtype']==np.int32
    tf_node['ak_attr']['shape']=shape['tf_attr']['value']
    tf_node['input']=tf_node['input'][:1]

def parse_Act(tf_node, graph):
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Activation'
    if tf_node['type'] =='Relu':
        tf_node['ak_type']='Relu'
        tf_node['ak_attr']['type']='Relu'
    else:
        raise Exception('un handel activation '+str(tf_node['type']))

def parse_Pooling(tf_node, graph):
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Pooling'
    if tf_node['type'] == 'MaxPool':
        ak_attr=tf_node['ak_attr']
        ak_attr['type'] = 'MAX'
        input_node = graph[tf_node['input'][0]]
        output_node = tf_node

        strides=tf_node['tf_attr']['strides']
        ksizes = tf_node['tf_attr']['ksize']
        padding =tf_node['tf_attr']['padding']
        dilations=tf_node['tf_attr'].get('dilations')
        data_format=tf_node['tf_attr']['data_format'].decode()

        kernel_shape=trans_move_attr(ksizes,data_format)
        strides=trans_move_attr(strides,data_format)

        padding=cal_padding(padding,kernel_shape,strides,dilations,data_format,input_node['out_shape'],output_node['out_shape'])

        ak_attr['window'] = kernel_shape
        ak_attr['padding'] = padding
        ak_attr['strides'] = strides



def cal_padding(padding,kernel_shape,strides,dilations,data_format,input_shape,output_shape,spatial=2):
    pads=[0]*spatial
    if type(padding)==bytes:
        padding=padding.decode()
        if dilations is None:
            dilations = [1] * spatial * 2
        if padding == 'SAME':
            pads = [0] * spatial
            if data_format=='NHWC':

                input_shape = spatial_map(input_shape, NHWC_TO_NCHW)
                output_shape = spatial_map(output_shape, NHWC_TO_NCHW)
            elif data_format!='NCHW':
                raise Exception('not suppor format '+data_format)
            for i in range(spatial):
                pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                pad = max(pad, 0)
                # pads[i] = pad // 2
                # pads[i + spatial] = pad - pad // 2
                pads1 = pad // 2
                pads2 = pad - pad // 2
                assert pads1==pads2,'pad in two dirctions must equal'
                pads[i] = pads1
            return pads
        elif padding == 'VALID':
            return pads
        else:
            raise ValueError("invalid padding value: " + padding)

def get_const_from_biinput(inputs,graph):
    assert len(inputs)<=2
    if graph[inputs[0]]['type']=='Const':
        return graph[inputs[0]]['tf_attr']['value']
    elif len(inputs)==2 and graph[inputs[1]]['type']=='Const':
        return graph[inputs[1]]['tf_attr']['value']
    return None

def get_bias(tf_node,graph):
    bias_weight = None
    output=tf_node['output']
    if len(output) == 1  and (graph[output[0]]['type'] == 'Add' or graph[output[0]]['type'] == 'BiasAdd'):
        add_node = graph[tf_node['output'][0]]
        bias_weight = get_const_from_biinput(add_node['input'], graph)

        if bias_weight is not None:
            add_node['visted'] = True
            tf_node['output']=add_node['output']

    return bias_weight

def parse_Conv2D(tf_node,graph):
    tf_node['visted'] = True
    tf_node['ak_type'] = 'Convolution'
    weights_node=graph[tf_node['input'][1]]

    assert weights_node['type'] == 'Const'
    weights = weights_node['tf_attr']['value']
    weights=weights.transpose((3,2,0,1))
    input_node = graph[tf_node['input'][0]]
    output_node = tf_node

    tf_attr=tf_node['tf_attr']
    data_format=tf_attr['data_format'].decode()

    padding=tf_attr['padding']
    dilations=trans_move_attr(tf_attr['dilations'],data_format)
    strides=trans_move_attr(tf_attr['strides'],data_format)
    kernel_shape=weights_node['out_shape'][:2]

    padding=cal_padding(padding,kernel_shape,strides,dilations,data_format,input_node['out_shape'],output_node['out_shape'])

    ak_attr=tf_node['ak_attr']
    ak_attr['weights']=weights
    ak_attr['padding'] = padding
    ak_attr['dilations']=dilations
    ak_attr['strides'] = strides
    ak_attr['bias_weights']=get_bias(tf_node,graph)
    ak_attr['group']=1




def parse_MatMul(tf_node,graph):
    tf_node['visted'] = True
    tf_node['ak_type']='Dense'
    out_name = tf_node['output'][0]
    in_name_0 = tf_node['input'][0]
    in_name_1 = tf_node['input'][1]
    in_type_0=graph[in_name_0]['type']
    in_type_1 = graph[in_name_1]['type']
    if in_type_0=='Const' and in_type_1=='Const':
        raise Exception('Whate hannpend both const')
    elif in_type_1=='Const' and tf_node['tf_attr']['transpose_a'] != True:
        weights=graph[in_name_1]['tf_attr']['value']
        if tf_node['tf_attr']['transpose_b']:
            weights=weights.T
        tf_node['ak_attr']['weights']=weights
        tf_node['input']=[tf_node['input'][0]]
    elif in_type_0=='Const' and tf_node['tf_attr']['transpose_b'] != True:
        weights = graph[in_name_1]['tf_attr']['value'].T
        if tf_node['tf_attr']['transpose_a']:
            weights=weights.T
        tf_node['ak_attr']['weights'] = weights
        tf_node['input'] = [tf_node['input'][1]]
    else:
        raise Exception('can`t parse '+str(tf_node))
    tf_node['ak_attr']['bias_weights']=get_bias(tf_node,graph)

