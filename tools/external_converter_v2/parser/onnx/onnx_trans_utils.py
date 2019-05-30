import onnx
import numpy as np
import math
from google.protobuf import text_format
from med_graph import MedNodeUtil, MedGraphUtil

ONNX_TO_ANAKIN_DTYPE = {
    1: np.float32,
    6: np.int32,
    7: np.int64,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}

ANAKIN_VALID_ATTRIBUTES = {
    'p', 'bias', 'axes', 'pads', 'mean', 'activation_beta',
    'spatial_scale', 'broadcast', 'pooled_shape', 'high', 'activation_alpha',
    'is_test', 'hidden_size', 'activations',
    'beta', 'input_as_shape', 'drop_states', 'alpha',
    'momentum', 'scale', 'axis', 'dilations', 'transB', 'axis_w', 'blocksize',
    'output_sequence', 'mode', 'perm',
    'min', 'seed', 'ends', 'paddings', 'to', 'gamma', 'width_scale',
    'normalize_variance', 'group', 'ratio', 'values',
    'dtype', 'output_shape', 'spatial', 'split', 'input_forget', 'keepdims', 'transA',
    'auto_pad', 'border', 'low', 'linear_before_reset', 'height_scale', 'output_padding',
    'shape', 'kernel_shape', 'epsilon', 'size', 'starts',
    'direction', 'max', 'clip', 'across_channels', 'value', 'strides',
    'extra_shape', 'scales', 'k', 'sample_size',
    'blocksize', 'epsilon', 'momentum'
}

def get_onnx_tensor_data(tensor):
    """
    Get data from tensor
    :param tensor:
    :return:
    """
    assert isinstance(tensor, onnx.TensorProto)
    is_raw = False
    # print 'tensor', tensor
    # tensor has raw_data and other_data
    if tensor.float_data is not None and len(tensor.float_data) > 0:
        data = tensor.float_data
        is_raw = False
    elif tensor.int32_data is not None and len(tensor.int32_data) > 0:
        data = tensor.int32_data
        is_raw = False
    elif tensor.string_data is not None and len(tensor.string_data) > 0:
        data = tensor.string_data
        is_raw = False
    elif tensor.int64_data is not None and len(tensor.int64_data) > 0:
        data = tensor.int64_data
        is_raw = False
    elif tensor.double_data is not None and len(tensor.double_data) > 0:
        data = tensor.double_data
        is_raw = False
    elif tensor.uint64_data is not None and len(tensor.uint64_data) > 0:
        data = tensor.uint64_data
        is_raw = False
    elif tensor.raw_data is not None and len(tensor.raw_data) > 0:
        data = tensor.raw_data
        is_raw = True
    else:
        print ('Error: ', tensor)
        exit(0)
    # da = np.array(data)
    # print da
    if tensor.data_type == 1: #FLOAT
        dtype = 'float32'
    elif tensor.data_type == 6: #INT32
        dtype = 'int32'
    elif tensor.data_type == 7: #INT64
        dtype = 'int64'
    elif tensor.data_type == 8: #string
        dtype = 'string'
    elif tensor.data_type == 11:  # string
        dtype = 'double'
    elif tensor.data_type == 12: #uint32
        dtype = 'uint32'
    elif tensor.data_type == 13: #uint32
        dtype = 'uint64'
    else:
        print ('Error: ', tensor.data_type)
        exit(0)
    return [is_raw, data, dtype]

def map_onnx_dtype(dtype):
    """
    :param dtype:
    :return:
    """
    return ONNX_TO_ANAKIN_DTYPE.get(dtype)

def has_key(attr, key_name):
    """
    dict key
    :param attr:
    :param key_name:
    :return:
    """
    for it in attr.keys():
        if it == key_name:
            return True

    return False

def onnx_to_anakin_tensor(tensor):
    """
    Convert onnx tensor to anakin med tensor
    :param tensor:
    :return:
    """
    # print tensor
    shape = []
    for dim in tensor.dims:
        shape.append(int(dim))
    # print('--shape: ', shape)
    [is_raw, data, dtype] = get_onnx_tensor_data(tensor)
    # print 'shape: ', shape
    # print 'is_raw: ', is_raw
    #print 'float_data: ', tensor.float_data
    # print(type(data),data,tensor.dtype,is_raw)
    if is_raw:
        if len(shape) > 0:
            # print 'type: ', tensor.data_type
            # print 'data: ', len(data)
            # print 'dtype: ', map_onnx_dtype(tensor.data_type)
            anakin_tensor = np.frombuffer(data, map_onnx_dtype(tensor.data_type))
            # print 'last len: ', len(anakin_tensor), anakin_tensor.shape
            # print 'shape: ', shape
            anakin_tensor = anakin_tensor.reshape(shape)
            # print 'last len: ', len(anakin_tensor), anakin_tensor.shape
            # exit(0)
        else:
            anakin_tensor = np.zeros(0)
        #print 'anakin_tensor: ', anakin_tensor
        # print('dtype: ', tensor.name, dtype, anakin_tensor.dtype)
        return anakin_tensor, shape, dtype
    else:
        #print 'data'
        return np.array(data).astype(map_onnx_dtype(tensor.data_type)), shape, dtype

def trans_const_node(node, weights):
    """
    trans const input to weight tensor
    :param node:
    :param weights:
    :return:
    """
    if len(node['input']) > 0:
        in_name = node['input'][0]
        weights_data = {}
        if in_name in weights:
            weights_node = weights[in_name]
            # print ('weights_node: ', node['name'], weights_node['shape'], weights_node['dtype'])
            if node['type'] == 'Reshape':
                shape_name = node['input'][1]
                if shape_name in weights:
                    shape_node = weights[shape_name]
                    shape = shape_node['data']
                    weights_data['shape'] = shape
                    weights_data['data'] = weights_node['data'].reshape(shape)
                    weights_data['dtype'] = weights_node['dtype']
                    # print ('weights_data: ', node['name'], weights_data['shape'], weights_data['dtype'])
                else:
                    print('Mul can not find shape_node', shape_name)
                    return None
            elif node['type'] == 'Unsqueeze': # axes = [1,2] [64] -> [64, 1, 1]
                axes = node['onnx_attr']['axes']
                shape = weights_node['shape'] # default nchw
                new_shape = []
                new_shape += shape
                num = len(shape)
                for i in axes:
                    if i >= num:
                        new_shape.append(1)
                # print ('shape: ', shape)
                # print ('new_shape: ', new_shape)
                weights_data['shape'] = new_shape
                weights_data['data'] = weights_node['data'].reshape(new_shape)
                weights_data['dtype'] = weights_node['dtype']
            elif node['type'] == 'Squeeze': # axes = [1,2] [1, 64, 1, 1] -> [1,64]
                axes = node['onnx_attr']['axes']
                shape = weights_node['shape'] # default nchw
                new_shape1 = shape
                new_shape = []
                num = len(shape)
                if num >= 1:
                    for i in range(0, num):
                        if i in axes:
                            new_shape1[i] = 0
                    for i in range(0, num):
                        if new_shape1[i] is not 0:
                            new_shape.append(new_shape1[i])
                else:
                    return None
                weights_data['shape'] = new_shape
                weights_data['data'] = weights_node['data'].reshape(new_shape)
                weights_data['dtype'] = weights_node['dtype']
            else:
                weights_data = weight_node
            node['visited'] = True
        else:
            print('Mul can not find input_node', in_name)
            return None
        # weights_data['shape'] = weights_data['shape'].astype(np.float32)
        return weights_data
    else:
        print('this node does not have input', node['name'])
        return None

def get_bias(node, weights, graph):
    """
    search graph find const input and the next op_type is Add, then convert the node to bias
    :param node:
    :param weights:
    :param graph:
    :return:
    """
    outs = node['output']
    output0 = graph[outs[0]]
    bias_node = None
    if len(outs) == 1 and output0['type'] == 'Add':
        ins = output0['input']
        for i in range(0, len(ins)):
            optype = graph[ins[i]]['type']
            if optype == 'Reshape' or optype == 'Unsqueeze' or optype == 'Squezze':
                bias_node = trans_const_node(graph[ins[i]], weights)
                if bias_node is not None:
                    #delete Add node
                    MedNodeUtil.redirecto_outputs_input_to_this(output0, graph, node['name'])
                    node['output'] = output0['output']
                    graph.pop(output0['name'])
                    #delete bias node
                    graph.pop(ins[i])
    return bias_node

def fusion_normL2_node(node_a, node_b, node_c, node, graph):
    """
    A->node_a->node->node_b->node_c->B
    A->node_c->B
    fusion: A->node->B
    :param node_a:
    :param node_b:
    :param node_c:
    :param node:
    :param graph:
    :return:
    """
    # print("node delete before: ", node['input'], node['output'])
    #first delete edge A->node_c
    top_in = node_a['input']
    A = graph[top_in[0]]
    # print('A delete before: ', A['output'])
    for ou in A['output']:
        if ou == node_c['name']:
            A['output'].remove(ou)
            break
    # print('A delete after: ', A['output'])
    B = node_c['output']
    #change node output
    # print('B delete before: ', graph[B[0]]['input'])
    node['output'] = B
    ins = graph[B[0]]['input']
    for i in range(0, len(ins)):
        if ins[i] == node_c['name']:
            ins[i] = node['name']
            # graph[B[0]]['input'].remove(ins)
            # graph[B[0]]['input'].append(node['name'])
    # print('B delete after: ', graph[B[0]]['input'])
    #delete node_b and node_c
    graph.pop(node_b['name'])
    graph.pop(node_c['name'])
    #change node input
    # print('A delete before: ', A['output'])
    node['input'] = node_a['input']
    A['output'] = node_a['output']
    # print('A delete after: ', A['output'])
    #delete node_a
    graph.pop(node_a['name'])
    # print("node delete after: ", node['input'], node['output'])

def fusion_PixelShuffle(node, out_node, outs, weights, graph):
    """
    node->out_node->transpose->reshape->B
    node->outs[0]->...->reshape->B
    node->outs[1]->...->reshape->B
    fusion: node->op_pixelshuffle->B
    :param node:
    :param out_node:
    :param outs:
    :param weights:
    :param graph:
    :return:
    """
    # print ('fusion_PixelShuffle begin: ')
    # print('node: ', node['name'], node['type'], node['input'], node['output'])
    # aaa = graph[node['output'][0]]
    # print('output: ', aaa['name'], aaa['type'], aaa['input'], aaa['output'])
    for ou in outs:
        if ou is not out_node['name']:
            if graph[ou]['type'] == 'Shape':
                continue
            else:
                print('Error Pattern: ', outs)
                return
    out_a = out_node['output']
    if len(out_a) == 1:
        out_b = graph[out_a[0]]
        if out_b['type'] == 'Transpose' and len(out_b['output']) == 1:
            out_name = out_b['output'][0]
            out_data = graph[out_name]
            if out_data['type'] == 'Reshape':
                out_list = [out_node['name'], out_name]
                for name in outs:
                    # print ('name: ', name)
                    if name not in out_list:
                        if graph.has_key(name) is not True:
                            continue
                        node1 = graph[name]
                        list_tmp = []
                        for name_a in node1['output']:
                            if graph.has_key(name_a) is not True:
                                if len(node1['output']) == 1:
                                    graph.pop(node1['name'])
                                    break
                            graph[name_a]['input'] = [node['name']]
                            list_tmp.append(name_a)
                        #outs.remove(name)
                        out_list.append(name)
                        if graph.has_key(name):
                            graph.pop(name)
                        # print ('remove name: ', name)
                        outs += list_tmp
                        # print ('delete output: ', out_list)
                node['output'] = [out_name]
                #delete Transpose and out_node
                graph.pop(out_b['name'])
                graph.pop(out_node['name'])
                out_data['type'] = 'PixelShuffle'
                scale_factor = 1
                if node['input'][1] in weights:
                    wei_shape = weights[node['input'][1]]['shape']
                    if len(wei_shape) == 4:
                        num = (wei_shape[0] / wei_shape[1])
                        sq = int(math.sqrt(num))
                        if num == sq * sq:
                            scale_factor = sq
                        else:
                            print('Error shape, it does not meet a*a = wei_shape[0] / wei_shape[1]', wei_shape[0], wei_shape[1])
                            exit(0)
                    else:
                        print('input is not right', node['input'])
                        exit(0)
                else:
                    print('weigths is not right', node['input'][1], wei_shape)
                out_data['onnx_attr'] ['scale_factor'] = scale_factor
                out_data['visited'] = True
                out_data['ak_type'] = 'PixelShuffle'
                out_data['ak_attr']['scale_factor'] = scale_factor
                out_data['ak_attr']['rw'] = scale_factor
                out_data['ak_attr']['rh'] = scale_factor
                if 'channel_first' in out_data['onnx_attr']:
                    out_data['ak_attr']['channel_first'] = out_data['onnx_attr']['channel_first']
                else:
                    out_data['ak_attr']['channel_first'] = True
            else:
                print('Error type ', out_data['name'], out_data['type'])
                exit()
        else:
            print('Error type ', out_b['name'], out_b['type'])
            exit()
    else:
        print('Error output lists ', out_a)
        exit()
    # print ('fusion_PixelShuffle after: ')
    # print('node: ', node['name'], node['type'], node['input'], node['output'])
    # aaa = graph[node['output'][0]]
    # print('output: ', aaa['name'], aaa['type'], aaa['input'], aaa['output'])

def delete_extra_node(node_a, node_b, node_c, graph):
    """
    A->node_a->..->B1->C1->node_c
    A->node_b->..->B2->C1->node_c
    A->node_c
    delete extra node: A->node_c->D
    :param node_a:
    :param node_b:
    :param node_c:
    :param node:
    :param graph:
    :return:
    """
    outs = node_c['name']
    while node_a['output'][0] is not outs:
        out_node = node_a['output'][0]
        a = graph[out_node]['output']
        if graph.has_key(a[0]) is not True:
            graph.pop(out_node)
            break
        graph[a[0]]['input'] = [node_a['name']]
        node_a['output'] = a
        graph.pop(out_node)
        # print('delete node: ', out_node)
    while node_b['output'][0] is not outs:
        # print (node_b['name'], node_b['output'])
        out_node = node_b['output'][0]
        b = graph[out_node]['output']
        if graph.has_key(b[0]) is not True:
            graph.pop(out_node)
            break
        graph[b[0]]['input'] = [node_b['name']]
        node_b['output'] = b
        graph.pop(out_node)
    graph.pop(node_a['name'])
    graph.pop(node_b['name'])

def parse_Div(onnx_node, weights, graph):
    """
    # Compute Y = normal_l2
    parse Div to Normalize
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Normalize'
    input_node = onnx_node['input']
    assert len(input_node) == 2
    input0 = input_node[0]
    input1 = input_node[1]

    # print 'input0', input0
    # print 'graph', graph
    in0_node = graph[input0]
    in1_node = graph[input1]
    const_node = {}
    in_node = {}
    # print in0_node
    # print in1_node
    if in0_node['type'] == 'Constant':
        #find the top node
        const_node = in0_node
        in_node = in1_node
    elif in1_node['type'] == 'Constant':
        #find the top node
        const_node = in1_node
        in_node = in0_node
    else:
        return
    top_node = in_node
    bot_node = graph[onnx_node['output'][0]]
    if top_node['type'] == 'ReduceL2':
        op_type = bot_node['type']
        if op_type == 'Unsqueeze' or op_type == 'Constant':
            bot_next_node = graph[bot_node['output'][0]]
            if bot_next_node['type'] == 'Div':
                ak_attr = onnx_node['ak_attr']
                ak_attr['begin_norm_axis'] = top_node['onnx_attr']['axes'][0]
                ak_attr['is_across_spatial'] = False
                ak_attr['is_shared_channel'] = True
                ak_attr['eps'] = 1e-6
                ak_attr['p'] = 2
                weights_node = {}
                weights_node['shape'] = [1]
                weights_node['data'] = [np.sqrt(bot_next_node['shape'][1])]#np.array(np.sqrt(bot_next_node['shape'][1])).astype(np.float32)
                weights_node['dtype'] = 'float32'
                ak_attr['weights'] = weights_node
                # delete node
                fusion_normL2_node(top_node, bot_node, bot_next_node, onnx_node, graph)
                ous = onnx_node['output']
                if len(ous) == 3:
                    if graph[ous[0]]['type'] == 'Reshape':
                        #change node
                        if graph[ous[1]]['type'] == 'Shape' and graph[ous[2]]['type'] == 'Shape':
                            #reshape
                            node_re = graph[ous[0]]
                            node_re['visited'] = 'True'
                            node_re['ak_type'] = 'Reshape'
                            ak_shape = node_re['ak_attr']
                            ak_shape['shape'] = [1, 128]
                            node_next = graph[node_re['output'][0]]
                            if node_next['input'][0] == node_re['name']:
                                wei_name = node_next['input'][1]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            else:
                                wei_name = node_next['input'][0]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            # print 'Reshape------: ', node_re['name'], node_re['ak_type'], node_re['shape']
                            #delete node
                            delete_extra_node(graph[ous[1]], graph[ous[2]], node_re, graph)
                            node_re['input'] = [onnx_node['name']]
                            onnx_node['output'] = [node_re['name']]
                    elif graph[ous[1]]['type'] == 'Reshape':
                        #change node
                        if graph[ous[0]]['type'] == 'Shape' and graph[ous[2]]['type'] == 'Shape':
                            #reshape
                            node_re = graph[ous[1]]
                            node_re['visited'] = 'True'
                            node_re['ak_type'] = 'Reshape'
                            ak_shape = node_re['ak_attr']
                            ak_shape['shape'] = [1, 128]
                            node_next = graph[node_re['output'][0]]
                            if node_next['input'][0] == node_re['name']:
                                wei_name = node_next['input'][1]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            else:
                                wei_name = node_next['input'][0]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            #delete node
                            delete_extra_node(graph[ous[1]], graph[ous[2]], node_re, graph)
                            node_re['input'] = [onnx_node['name']]
                            onnx_node['output'] = [node_re['name']]
                    elif graph[ous[2]]['type'] == 'Reshape':
                        #change node
                        if graph[ous[0]]['type'] == 'Shape' and graph[ous[1]]['type'] == 'Shape':
                            #reshape
                            node_re = graph[ous[2]]
                            node_re['visited'] = 'True'
                            node_re['ak_type'] = 'Reshape'
                            ak_shape = node_re['ak_attr']
                            ak_shape['shape'] = [1, 128]
                            node_next = graph[node_re['output'][0]]
                            if node_next['input'][0] == node_re['name']:
                                wei_name = node_next['input'][1]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            else:
                                wei_name = node_next['input'][0]
                                wshape = weights[wei_name]['shape']
                                ak_shape['shape'] = [1, wshape[0]]
                            #delete node
                            delete_extra_node(graph[ous[1]], graph[ous[2]], node_re, graph)
                            node_re['input'] = [onnx_node['name']]
                            onnx_node['output'] = [node_re['name']]

            else:
                print('Error: ', in_node['type'])
                exit(0)
        else:
            print('Error Pattern: ', in_node['type'])
            # exit(0)
    else:
        print('Error Pattern: ', in_node['type'])
        # exit(0)

def rm_weight_node(onnx_node, weights, graph):
    """
    remove weights node
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    for node in onnx_node.keys():
        in_node = onnx_node[node]['input']
        for name in in_node:
            if weights.has_key(name):
                in_node.remove(name)

def parse_Conv(onnx_node, weights, graph):
    """
    parse conv
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
   #print 'parse_Conv2D'
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Convolution'
    wei_name = onnx_node['input'][1]
    weights_node = weights[wei_name]
    if weights.has_key(wei_name):
        weights_node = weights[wei_name]
    else:
        print ('conv can not find weights', wei_name)
    #assert weights_node['type'] == 'Const'
    weights_data = weights_node

    #print 'weights: ', weights_data
    #exit()
    bias_node = None
    if len(onnx_node['input']) > 2:
        bias_name = onnx_node['input'][2]
        bias_node = weights[bias_name]
        if weights.has_key(bias_name):
            bias_node = weights[bias_name]
        else:
            print ('conv can not find bias', bias_name)
        '''
        print 'bias dtype', bias_node['dtype']
        print 'bias shape ', bias_node['shape']
        print 'bias data', bias_node['data']
        exit()
        '''
        onnx_node['input'].remove(bias_name)

    onnx_attr = onnx_node['onnx_attr']
    group = 1
    if 'group' in onnx_attr.keys():
        group = onnx_attr['group']

    padding_val = []
    if 'pads' in onnx_attr.keys():
        #print 'pads: ', type(onnx_attr['pads'][0])
        padding_val = onnx_attr['pads'] #T L B R
        if len(onnx_attr['pads']) == 1:
            padding_val = [0, onnx_attr['pads'][0]]
    else:
        padding_val = [0, 0]

    dilations = []
    if 'dilations' in onnx_attr.keys():
        dilations = onnx_attr['dilations']
        if len(onnx_attr['dilations']) == 1:
            dilations = [1, onnx_attr['dilations'][0]]
    else:
        dilations = [1, 1]

    strides = []
    if 'strides' in onnx_attr.keys():
        strides = onnx_attr['strides']
        if len(onnx_attr['strides']) == 1:
            strides = [1, onnx_attr['strides'][0]]
    else:
        strides = [1, 1]

    kernel_shape = onnx_attr['kernel_shape']

    if len(onnx_attr['kernel_shape']) == 1:
        chin = weights_data['shape'][1]
        # print '**shape**', weights_data['shape'], type(chin), type(strides[0])
        kernel_shape = [1, onnx_attr['kernel_shape'][0]]
    #padding deal include padding
    if 'auto_pad' in onnx_attr.keys(): #onnx_attr['auto_pad'] == 'SAME_LOWER' or onnx_attr['auto_pad'] == 'SAME_UPPER':
       #out_shape[2] = ceil((in_shape[2]- kernel_h) / stride_h)
       #pad[0] = (out_shape[2] - 1) * stride_h + \ kernel_h - in_shape[2]
       padding = [1, 1]
    padding = [padding_val[0], padding_val[1]]

    ak_attr = onnx_node['ak_attr']
    ak_attr['weights'] = weights_data
    ak_attr['padding'] = padding
    ak_attr['dilations'] = dilations
    ak_attr['strides'] = strides
    ak_attr['kernel'] = kernel_shape
    ak_attr['group'] = group
    if bias_node is not None:
        ak_attr['bias'] = bias_node

    # pixelShuffle
    if len(onnx_node['output']) == 5:
        outs = onnx_node['output']
        for i in range(0, len(outs)):
            if graph[outs[i]]['type'] == 'Reshape':
                fusion_PixelShuffle(onnx_node, graph[outs[i]], outs, weights, graph)
                # refind_node_delete(onnx_node, graph)
                break

    inputs = onnx_node['input']
    inputs.remove(wei_name)
    '''
    for name in inputs:
        if name == wei_name:
            inputs.remove(name)
        if name == bias_name:
            inputs.remove(bias_name)
    '''

def parse_Mul(onnx_node, weights, graph):
    """
    # Compute Y = A * B + C
    parse Mul to dense
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Scale'
    input_node = onnx_node['input']
    input0 = input_node[0]
    input1 = input_node[1]
    in0_type = graph[input0]['type']
    in1_type = graph[input1]['type']
    weights_node = {}
    if in0_type == 'Reshape' or in0_type == 'Unsqueeze' or in0_type == 'Squezze':
        weights_node = trans_const_node(graph[input0], weights)
        if weights_node is not None:
            # remove the input node
            graph.pop(input0)
            onnx_node['input'].remove(input0)
            # onnx_node['input'].remove(wei_name)
        else:
            print ('MUL can not find weights', input0)
            exit(0)
    elif in1_type == 'Reshape' or in1_type == 'Unsqueeze' or in1_type == 'Squezze':
        weights_node = trans_const_node(graph[input1], weights)
        if weights_node is not None:
            # remove the input node
            graph.pop(input1)
            onnx_node['input'].remove(input1)
        else:
            print ('can not find weights', input1)
            exit(0)
    elif in0_type == 'Constant' or in1_type == 'Constant':
        weights_node = {}
        '''
        node = graph[onnx_node['input'][0]]
        wei_name = node['input'][1]
        a = weights[wei_name]['shape'][0]
        '''
        weights_node['shape'] = [64] #[a]
        data = np.ones(weights_node['shape'])
        if 'broadcast' in onnx_node['onnx_attr']:
            for i in range(0, weights_node['shape'][0]):
                data[i] = onnx_node['onnx_attr']['broadcast'] # 1
            weights_node['data'] = data
            weights_node['dtype'] = "float32"
            if in0_type == 'Constant':
                # print('input0: ', input0)
                graph.pop(input0)
                onnx_node['input'].remove(input0)
            else:
                # print('input1: ', input1)
                graph.pop(input1)
                onnx_node['input'].remove(input1)
        else:
            print ('Mul parse Error')
            exit(0)
    else:
        print ('Mul parse Error Pattern: ', in0_type, in1_type)
        # return
        # exit(0)
    ak_attr = onnx_node['ak_attr']
    ak_attr['weights'] = weights_node
    bias_node = get_bias(onnx_node, weights, graph)
    if bias_node is not None:
        ak_attr['bias'] = bias_node

def parse_Gemm(onnx_node, weights, graph):
    """
    # Compute Y = alpha * A' * B' + beta * C
    parse Gemm to dense
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Dense'

    onnx_attr = onnx_node['onnx_attr']
    alpha = 1.0
    if 'alpha' in onnx_attr.keys():
        alpha = onnx_attr['alpha']

    beta = 1.0
    if 'beta' in onnx_attr.keys():
        beta = onnx_attr['beta']

    transA = 0
    if 'transA' in onnx_attr.keys():
        transA = onnx_attr['transA']
    else:
        transA = 0

    transB = 0
    if 'transB' in onnx_attr.keys():
        transB = onnx_attr['transB']
    else:
        transB = 0

    wei_name = onnx_node['input'][1]
    weights_node = {}
    if weights.has_key(wei_name):
        weights_node = weights[wei_name]
        # onnx_node['input'].remove(wei_name)
    else:
        node = graph[wei_name]
        weights_node = trans_const_node(node, weights)
        if weights_node is not None:
            # remove the input node
            graph.pop(wei_name)
            # onnx_node['input'].remove(wei_name)
        else:
            print ('Gemm can not find weights', wei_name)
            exit(0)
    #assert weights_node['type'] == 'Const'
    # weights_data = weights_node

    ak_attr = onnx_node['ak_attr']
    if beta == 1:
        if len(onnx_node['input']) > 2:
            bias_name = onnx_node['input'][2]
            # bias_node = weights[bias_name]
            if weights.has_key(bias_name):
                bias_node = weights[bias_name]
            else:
                bias_node = graph[bias_name]
                print ('Gemm can not find bias', bias_name)
            # print('Dense input: ', onnx_node['input'])
            onnx_node['input'].remove(bias_name)
            # print('Dense input: ', onnx_node['input'])
            ak_attr['bias'] = bias_node

    #print 'name: ', onnx_node['name']
    #print 'shape', weights_data['shape']
    if alpha == 0 or transA == 1:
        ak_attr['weights'] = None
        ak_attr['Gemm'] = 0
        print ('Gemm Error, alpha, transA', alpha, transA)
        exit(0)
    else:
        weights_data = {}
        if transB == 1:
           #print 'trans'
            ak_attr['trans'] = 1
            # print ('trans before: ', weights_node['shape'])
            # weights_data['data'] = np.transpose(weights_node['data'])
            # weights_data['shape'] = [weights_node['shape'][1],  weights_node['shape'][0]]
            # weights_data['dtype'] = weights_node['dtype']
            # print ('trans after: ', weights_data['shape'])
        else:
            ak_attr['trans'] = 0
            # weights_data = weights_node
        weights_data = weights_node
        ak_attr['weights'] = weights_data
        ak_attr['Gemm'] = 1
    #ak_attr['out_dim'] = weights_data
    onnx_node['input'].remove(wei_name)

def parse_Act(onnx_node, weights, graph):
    """
    parse Act
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Activation'
    if onnx_node['type'] == 'Relu':
        onnx_node['ak_type'] = 'ReLU'
        onnx_node['ak_attr']['type'] = 'Relu'
    elif onnx_node['type'] == 'LeakyRelu':
        # onnx_node['ak_type'] = 'PReLU'
        onnx_node['ak_attr']['type'] = 'PReLU'
        onnx_attr = onnx_node['onnx_attr']
        slope = 0.01
        if 'alpha' in onnx_attr:
            slope = onnx_attr['alpha']
        weights_node = {}
        weights_node['dtype'] = 'float32'
        weights_node['shape'] = [1]
        weights_node['data'] = [slope]
        onnx_node['ak_attr']['weights'] = weights_node
        onnx_node['ak_attr']['channel_shared'] = True
    else:
        raise Exception('un handel activation ' + str(onnx_node.op_type))

def parse_Concat(onnx_node, weights, graph):
    """
    parse Concat
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Concat'
    onnx_attr = onnx_node['onnx_attr']
    ak_attr = onnx_node['ak_attr']
    if 'axis' in onnx_attr.keys():
        ak_attr['axis'] = onnx_attr['axis']
    else:
        ak_attr['axis'] = 0

def parse_Reshape(onnx_node, weights, graph):
    """
    parse Reshape
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Reshape'
    shape_name = onnx_node['input'][1]
    shape_node = {} #weights[shape_name]
    if weights.has_key(shape_name):
        shape_node = weights[shape_name]
    else:
        if len(onnx_node['input']) == 2:
            in_node0 = graph[onnx_node['input'][0]]
            in_node1 = graph[onnx_node['input'][1]]
            if in_node0['type'] == 'Constant':
                shape_node['data'] = in_node1['onnx_attr']['value'][0]
            elif in_node1['type'] == 'Constant':
                shape_node['data'] = in_node1['onnx_attr']['value'][0]
                # print shape_node, type(shape_node['data'])
            else:
                print ('Reshape can not find weights', shape_name)
                exit(0)
        else:
            shape_node['shape'] = [1,1,1,1]
            shape_node['data'] = [1]
            print ('Reshape can not find weights', shape_name)
            exit(0)

    ak_attr = onnx_node['ak_attr']
    # array = np.array(shape_node['shape'])
    data = shape_node['data']

    input_name = onnx_node['input'][0]

    shape = []
    if data[0] == 0:
        onnx_node['ak_type'] = 'Flatten'
        ak_attr['start_axis'] = 1
        ak_attr['end_axis'] = -1
        ak_attr['type'] = 'Flatten'
    else:
        if len(data) == 5:
            if data[0] == 1:
                shape = [data[1], data[2], data[3], data[4]]
            else:
                print ('Reshape does not support 5 dims ', data)
                exit()
        # elif len(data) > 5:
        #     print ('Reshape does not support >5 dims ', data)
        #     exit()
        else:
            shape = data

        ak_attr['type'] = 'Reshape'
    # print ('***Reshape:*** ', shape)
    ak_attr['shape'] = shape

    # print onnx_node['input']
    onnx_node['input'].pop(1)
    # print onnx_node['input']

def parse_Transpose(onnx_node, weights, graph):
    """
    parse Transpose to Permute
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Permute'

    ak_attr = onnx_node['ak_attr']
    data = onnx_node['onnx_attr']['perm']

    shape = []

    if len(data) == 5 and data[0] == 0:
        shape = [data[1]-1, data[2]-1, data[3]-1, data[4]-1]
    # elif len(data) >= 5:
    #     shape = data
    #     print ('Permute does not support 5 dims permute ', data)
    #     # exit(0)
    else:
        shape = data
    # print('data: ', data)
    # print('shape: ', shape)
    ak_attr['shape'] = shape

def parse_Add(onnx_node, weights, graph):
    """
    parse Add to Eltwise
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    assert len(onnx_node['input']) == 2

    ak_attr = onnx_node['ak_attr']
    onnx_node['ak_type'] = 'Eltwise'
    ak_attr['type'] = 'Add'

def parse_Sum(onnx_node, weights, graph):
    """
    parse Sum to Eltwise
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    assert len(onnx_node['input']) == 2

    ak_attr = onnx_node['ak_attr']
    onnx_node['ak_type'] = 'Eltwise'
    ak_attr['type'] = 'Add'

def parse_Pooling(onnx_node, weights, graph):
    """
    parse Pooling
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Pooling'
    ak_attr = onnx_node['ak_attr']
    onnx_attr = onnx_node['onnx_attr']

    padding_val = []
    if 'pads' in onnx_attr.keys():
        padding_val = onnx_attr['pads']
    else:
        padding_val = [0, 0]

    dilations = []
    if 'dilations' in onnx_attr.keys():
        dilations = onnx_attr['dilations']
    else:
        dilations = [1, 1]

    strides = []
    if 'strides' in onnx_attr.keys():
        strides = onnx_attr['strides']
    else:
        strides = [1, 1]

    kernel_shape = []
    if 'kernel_shape' in onnx_attr.keys():
        kernel_shape = onnx_attr['kernel_shape']
    else:
        kernel_shape = [1, 1]
    # padding deal inlcuding pading
    if 'auto_pad' in onnx_attr.keys(): #onnx_attr['auto_pad'] == 'SAME_LOWER' or onnx_attr['auto_pad'] == 'SAME_UPPER':
       #out_shape[2] = ceil((in_shape[2]- kernel_h) / stride_h)
       #pad[0] = (out_shape[2] - 1) * stride_h + \ kernel_h - in_shape[2]
       padding_val = [1, 1]
    # padding = [1, 1, 1, 1] =[top, left, bottom, right]
    # else:
    padding = [padding_val[0], padding_val[1]]
    if len(padding_val) == 4:
        a = padding_val[0] + padding_val[2]
        b = padding_val[1] + padding_val[3]
        pad_val0 = a / 2
        pad_val1 = b / 2
        # print 'padding:', pad_val0, pad_val1
        padding = [pad_val0, pad_val1]
        # inception v2
        # padding = [padding_val[2], padding_val[3]]


    ak_attr['window'] = kernel_shape
    ak_attr['padding'] = padding
    ak_attr['strides'] = strides

    if onnx_node['type'] == 'MaxPool':
        ak_attr['type'] = 'MAX'
        ak_attr['global_pooling'] = False

    if onnx_node['type'] == 'AveragePool':
        if 'count_include_pad'in onnx_attr.keys():
            ak_attr['type'] = 'AVG'
        else:
            ak_attr['type'] = 'AVGEXC'
        ak_attr['global_pooling'] = False
        # padding deal
        # if onnx_attr['atuo_pad'] == 'SAME_LOWER' or onnx_attr['atuo_pad'] == 'SAME_UPPER':
        #     padding = [0, 0]
        # else:
        #     padding = [padding_val[1], padding_val[0]]

    if onnx_node['type'] == 'GlobalMaxPool':
        ak_attr['type'] = 'MAX'
        ak_attr['global_pooling'] = True

        padding_val = [0, 0]
        strides = [0, 0]
        kernel_shape = [1, 1]

    if onnx_node['type'] == 'GlobalAveragePool':
        ak_attr['type'] = 'AVG'
        ak_attr['global_pooling'] = True

        padding_val = [0, 0]
        strides = [0, 0]
        kernel_shape = [1, 1]

    ak_attr['window'] = kernel_shape
    ak_attr['padding'] = padding #padding_val
    ak_attr['strides'] = strides

def parse_ImageScaler(onnx_node, weights, graph):
    """
    parse ImageScaler
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Scale'
    ak_attr = onnx_node['ak_attr']

    scale_val = onnx_node['onnx_attr']['scale']
    shape = [1, 1, 1, 3]
    scale_val = [1.0, 1.0, 1.0]
    if 'scale' in onnx_node['onnx_attr']:
        scale_val = onnx_node['onnx_attr']['scale']
    if type(scale_val) is 'float':
        scale_val =[ scale_val, scale_val, scale_val]
    scale_np = np.full(shape, scale_val) #np.arange([scale_val])
    weight_tensor = {}
    weight_tensor['shape'] = shape
    weight_tensor['data'] = scale_np
    weight_tensor['dtype'] = 'float32'
    ak_attr['weights'] = weight_tensor

    bias_val = [1.0]
    if 'bias' in onnx_node['onnx_attr']:
        bias_val = onnx_node['onnx_attr']['bias']
        # print 'bias: ', len(bias_val)
        shape_b = [len(bias_val)]
        # print 'shape_b: ', shape_b
        bias_tensor = {}
        bias_tensor['shape'] = shape_b
        bias_tensor['data'] = bias_val
        bias_tensor['dtype'] = 'float32'
        ak_attr['bias'] = bias_tensor


def parse_Dropout(onnx_node, weights, graph):
    """
    parse Dropout
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Scale'
    ak_attr = onnx_node['ak_attr']
    '''
    ratio   (float, default 0.5) the ratio of random dropout
    is_test (int) if nonzero, run dropout in test mode where the output is simply Y = X.
    '''
    if 'is_test' in onnx_node['onnx_attr'].keys():
        if onnx_node['onnx_attr']['is_test']  == 0:
            ak_attr['drop'] = 1 #Ydata[i] = Xdata[i] * scale * mask_data[i];
        else:
            ak_attr['drop'] = 0
            onnx_node['output'].pop(len(onnx_node['output'])-1) #delete mask_node
            print ('it not support, Error')
            return
    else:
        ak_attr['drop'] = 0
    scale_val = onnx_node['onnx_attr']['ratio']
    shape = [1, 1, 1, 1]
    scale_np = np.full(shape, scale_val) #np.arange([scale_val])
    weight_tensor = {}
    weight_tensor['shape'] = shape
    weight_tensor['data'] = scale_np
    weight_tensor['dtype'] = 'float32'
    ak_attr['weights'] = weight_tensor
    ak_attr['axis'] = 0
    ak_attr['num_axes'] = 0

def parse_Softmax(onnx_node, weights, graph):
    """
    parse sooftmax
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Softmax'
    if 'axis' in onnx_node['onnx_attr']:
        onnx_node['ak_attr']['axis'] = onnx_node['onnx_attr']['axis']
    else:
        onnx_node['ak_attr']['axis'] = 1

def parse_Lrn(onnx_node, weights, graph):
    """
    parse LRN
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'LRN'
    ak_attr = onnx_node['ak_attr']
    onnx_attr = onnx_node['onnx_attr']
    local_size = 0
    if 'size' in onnx_attr.keys():
        local_size = onnx_attr['size']
    alpha = 0.0001
    if 'alpha' in onnx_attr.keys():
        alpha = onnx_attr['alpha']
    beta = 0.75
    if 'beta' in onnx_attr.keys():
        beta = onnx_attr['beta']
    k = 1
    if 'bias' in onnx_attr.keys():
        k = onnx_attr['bias']
    ak_attr['local_size'] = local_size
    ak_attr['alpha'] = alpha / local_size
    ak_attr['beta'] = beta
    ak_attr['k'] = k

def parse_BatchNorm(onnx_node, weights, graph):
    """
    parse BatchNorm
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Scale'
    ak_attr = onnx_node['ak_attr']
    assert len(onnx_node['input']) == 5

    alpha_name = onnx_node['input'][1]
    beta_name = onnx_node['input'][2]
    mean_name = onnx_node['input'][3]
    var_name = onnx_node['input'][4]

    alpha_node = weights[alpha_name]
    if weights.has_key(alpha_name):
        alpha_node = weights[alpha_name]
    else:
        print ('BatchNorm can not find alpha_name', alpha_name)
        exit(0)
        return

    beta_node = weights[beta_name]
    if weights.has_key(beta_name):
        beta_node = weights[beta_name]
    else:
        print ('BatchNorm can not find beta_name', beta_name)
        exit(0)
        return

    mean_node = weights[mean_name]
    if weights.has_key(mean_name):
        mean_node = weights[mean_name]
    else:
        print ('BatchNorm can not find mean_name', mean_name)
        exit(0)
        return

    var_node = weights[var_name]
    if weights.has_key(var_name):
        var_node = weights[var_name]
    else:
        print ('BatchNorm can not find var_name', var_name)
        exit(0)
        return

    onnx_attr = onnx_node['onnx_attr']
    eps = 1e-5
    if 'epsilon' in onnx_attr.keys():
        eps = onnx_attr['epsilon']
    momentum = 0.9
    if 'momentum' in onnx_attr.keys():
        momentum = onnx_attr['momentum']
    spatial = 1
    if 'spatial' in onnx_attr.keys():
        spatial = onnx_attr['spatial']

    # print 'type: ', type(var_node['data'])
    var_data = np.array(var_node['data'])
    alpha_data = np.array(alpha_node['data'])
    beta_data = np.array(beta_node['data'])
    mean_data = np.array(mean_node['data'])
    var = np.sqrt(var_data.flatten() + eps)
    np_scale = alpha_data.flatten() / var
    np_bias = beta_data.flatten() - (alpha_data.flatten() * mean_data.flatten() / var)

    # ak_attr['weights'] = np_scale.astype('float32')
    # ak_attr['bias'] = np_bias.astype('float32')
    scale_tensor = {}
    bias_tensor = {}
    scale_tensor['dtype'] = 'float32'
    scale_tensor['data'] = np_scale
    scale_tensor['shape'] = np_scale.shape

    # print 'parse_BatchNorm scale: ', np_scale.shape

    bias_tensor['dtype'] = 'float32'
    bias_tensor['data'] = np_bias
    bias_tensor['shape'] = np_bias.shape

    # print 'parse_BatchNorm bias: ', np_bias.shape

    ak_attr['weights'] = scale_tensor
    ak_attr['bias'] = bias_tensor

    MedNodeUtil.retain_input(onnx_node, [onnx_node['input'][0]])

def parse_Slice(onnx_node, weights, graph):
    """
    parse Slice [axes, starts, ends]
    axes[0]==>[starts[0],ends[0]]
    axes[1]==>[starts[1],ends[1]]
    :param onnx_node:
    :param weights:
    :param graph:
    :return:
    """
    onnx_node['visited'] = True
    onnx_node['ak_type'] = 'Slice'
    ak_attr = onnx_node['ak_attr']
    onnx_attr = onnx_node['onnx_attr']
    ak_attr['axis'] = onnx_attr['axes']
    ak_attr['slice_point'] = onnx_attr['starts']
    ak_attr['slice_dim'] = onnx_attr['ends']
