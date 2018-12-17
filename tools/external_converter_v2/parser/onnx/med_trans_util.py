import numpy as np
from ..graph_io import TensorProtoIO, OpsProtoIO
from ..operations import OpsParam

def shape_2_ak_shape(shape):
    """
    onnx shape to anakin shape
    :param shape:
    :return:
    """
    mini_shape = [i for i in shape if (i is not None and i > 0)]
    return map(int, [1] * (4 - len(mini_shape)) + list(mini_shape))

def np_2_ak_tensor(np_tensor):
    """
    onnx np array to tensor
    :param np_tensor:
    :return:
    """
    data_type_map2 ={
        np.dtype('float32'): 'float',
        np.dtype('int32'): 'int',
        np.dtype('bool'): 'bool'
    }
    data_type_map = {
       'float32': 'float',
        'int32': 'int',
        'bool': 'bool'
    }
    # print 'np_tensor: ', np_tensor['dtype']
    #exit()
    type_str = data_type_map.get(np_tensor['dtype'])
    #assert type_str != None
    ak_tensor = TensorProtoIO()
    # print 'name: ', np_tensor['name']
    # print 'shape: ', np_tensor['shape']
    # print 'type_str: ', type_str
    # print 'np ', type(np_tensor['data'])
    # print 'np_type:', np_tensor['data'].dtype
    # print 'len:', len(np_tensor['data'])
    # print 'data: ', np_tensor['data']
    ak_tensor.set_shape(shape_2_ak_shape(np_tensor['shape']))
    # ak_tensor.set_data(np_tensor['data'], type_str)
    # print('type: ', type(np_tensor['data']), np_tensor['shape'], np_tensor['dtype'], type_str)
    if (len(np_tensor['shape']) == 1):
        ak_tensor.set_data(np_tensor['data'], type_str)
    else:
        ak_tensor.set_data(np_tensor['data'].flatten(), type_str)
    return ak_tensor


class MedTransAK:
    """
    tools on med graph to anakin graph
    """
    def __init__(self):
        self.input_count=0

    # def AssignDecorator(Converter):
    #     def warpper(self,ak_node, med_node):
    #         param = OpsParam()
    #         ak_op = OpsProtoIO()
    #         med_attr=med_node['ak_attr']
    #
    #         Converter(self, med_attr, param)
    #
    #         param.feed_node_attr(ak_node)
    #         ak_op.set_name(med_node['ak_type'])
    #         ak_node.set_op(ak_op())
    #         [ak_node.add_in(i) for i in med_node['input']]
    #         [ak_node.add_out(i) for i in med_node['output']]
    #
    #     return warpper

    def Convolution(self, med_attr, param):
        """
        get Conv param
        :param med_attr:
        :param param:
        :return:
        """
        np_filters = med_attr['weights']
        param.weight_1 = np_2_ak_tensor(np_filters)
        param.filter_num = np_filters['shape'][0] #?
        param.kernel_size = med_attr['kernel']
        param.strides = med_attr['strides']
        param.padding = med_attr['padding'] #T L B R
        param.dilation_rate = med_attr['dilations']
        # print('-------conv group----')
        # print('filter_num: ', param.filter_num)
        # print('group: ', med_attr['group'])
        param.group = med_attr['group']
        param.axis = 1
        if med_attr.get('bias') is not None:
            param.bias_term = True
            bias_tensor = med_attr['bias']
            bias_tensor['shape'] = [1, 1, 1, bias_tensor['shape'][-1]]
            param.weight_2 = np_2_ak_tensor(bias_tensor)
        else:
            param.bias_term = False

    def Normalize(self, med_attr, param):
        """
        get Normalize param
        :param med_attr:
        :param param:
        :return:
        """
        np_filters = med_attr['weights']
        param.weight_1 = np_2_ak_tensor(np_filters)
        param.begin_norm_axis = med_attr['begin_norm_axis']
        param.is_across_spatial = med_attr['is_across_spatial']
        param.is_shared_channel = med_attr['is_shared_channel'] #T L B R
        param.eps = med_attr['eps']
        param.p = med_attr['p']

    def Dense(self, med_attr, param):
        """
        get dense param
        :param med_attr:
        :param param:
        :return:
        """
        param.axis = 1
        param.out_dim = 0
        if med_attr['Gemm'] == 1:
            param.weight_1 = np_2_ak_tensor(med_attr['weights'])
            # if med_attr.get('trans') is not None:
            #     param.out_dim = med_attr['weights']['shape'][1]
            #     print'trans out_dim', param.out_dim, type(param.out_dim)
            # else:
            #     param.out_dim = med_attr['weights']['shape'][0]
                # print'out_dim', param.out_dim
        else:
            param.weight_1 = TensorProtoIO()

        if med_attr.get('bias') is not None:
            param.bias_term = True
            param.weight_2 = np_2_ak_tensor(med_attr['bias'])
            param.out_dim = len(med_attr['bias']['data'].flatten())
        else:
            param.bias_term = False
        #print 'shape: ', med_attr['weights']['shape']

    def ReLU(self, med_attr, param):
        """
        get relu param
        :param med_attr:
        :param param:
        :return:
        """
        if med_attr.get('alpha') is None:
            param.alpha = 0.0
        else:
            param.alpha = med_attr['type']

    def PReLU(self, med_attr, param):
        """
        get relu param
        :param med_attr:
        :param param:
        :return:
        """
        if med_attr.get('channel_shared') is None:
            param.channel_shared = False
        else:
            param.channel_shared = med_attr['channel_shared']

    def Concat(self, med_attr, param):
        """
        get concat param
        :param med_attr:
        :param param:
        :return:
        """
        if med_attr.get('axis') is None:
            param.axis = 0.0
        else:
            param.axis = med_attr['axis']

    def Activation(self, med_attr, param):
        """
        grt act param
        :param med_attr:
        :param param:
        :return:
        """
        param.type = med_attr['type']
        if med_attr['type'] == 'PReLU':
            if med_attr.get('channel_shared') is None:
                param.channel_shared = False
            else:
                param.channel_shared = med_attr['channel_shared']
            param.weight_1 = np_2_ak_tensor(med_attr['weights'])

    def Reshape(self, med_attr, param):
        """
        get reshape param
        :param med_attr:
        :param param:
        :return:
        """
        shape = med_attr['shape']
        if isinstance(shape, type(np.array([]))):
            shape = [int(i) for i in shape]
        # print('***Reshape:*** ', shape)
        param.dims = shape_2_ak_shape(shape)
        # print(param.dims)
        pass

    def Pooling(self, med_attr, param):
        """
        get pooling param
        :param med_attr:
        :param param:
        :return:
        """
        param.method = med_attr['type']
        param.pool_size = med_attr['window']
        param.strides = med_attr['strides']
        param.padding = med_attr['padding'] # T L B R
        if med_attr.get('global_pooling') is None:
            param.global_pooling = False
        else:
            param.global_pooling = med_attr['global_pooling']
        # if med_attr['padding'][0] == 0:
        #     param.cmp_out_shape_floor_as_conv = False
        # else:
        #     param.cmp_out_shape_floor_as_conv = True
        param.cmp_out_shape_floor_as_conv = True
        pass

    def Input(self, med_attr, param):
        """
        get input param
        :param med_attr:
        :param param:
        :return:
        """
        param.input_shape = shape_2_ak_shape(med_attr['shape'])
        param.alias = 'input_' + str(self.input_count)
        self.input_count += 1

    def Dropout(self, med_attr, param):
        """
        get dropoout param
        :param med_attr:
        :param param:
        :return:
        """
        param.ratio = med_attr['ratio']

    def Split(self, med_attr, param):
        """
        get split param
        :param med_attr:
        :param param:
        :return:
        """
        param.split_num = med_attr['split_num']

    def Eltwise(self, med_attr, param):
        """
        get eltwise param
        :param med_attr:
        :param param:
        :return:
        """
        assert med_attr['type'] == 'Add'
        param.type = med_attr['type']
        param.coeff = [1.0, 1.0]

    def Scale(self, med_attr, param):
        """
        get scale param
        :param med_attr:
        :param param:
        :return:
        """
        # print 'weights'
        param.weight_1 = np_2_ak_tensor(med_attr['weights'])
        # print 'bias'
        if med_attr.get('bias') is not None:
            param.weight_2 = np_2_ak_tensor(med_attr['bias'])
            param.bias_term = True
            param.axis = 1
            param.num_axes = 1
        else:
            param.bias_term = False
            param.axis = 0
            param.num_axes = 0

    def Flatten(self, med_attr, param):
        """
        get flatten param
        :param med_attr:
        :param param:
        :return:
        """
        param.start_axis = med_attr['start_axis']
        param.end_axis = med_attr['end_axis']

    def LRN(self, med_attr, param):
        """
        get lrn param
        :param med_attr:
        :param param:
        :return:
        """
        param.local_size = med_attr['local_size']
        param.alpha = med_attr['alpha']
        param.beta = med_attr['beta']
        param.k = med_attr['k']
        param.norm_region = "ACROSS_CHANNELS"

    def Softmax(self, med_attr, param):
        """
        get softmax param
        :param med_attr:
        :param param:
        :return:
        """
        if med_attr.get('axis') is None:
            param.axis = 3
        else:
            param.axis = med_attr['axis']
        pass

    def map_med_2_ak(self, ak_node, med_node):
        """
        med graph convert to anakin graph
        :param ak_node:
        :param med_node:
        :return:
        """
        type_name = med_node['ak_type']
        func = getattr(self, type_name, None)
        param = OpsParam()
        ak_op = OpsProtoIO()
        med_attr = med_node['ak_attr']
        #print type_name

        # print med_node['name'], med_node['type'], med_node['ak_type']
        func(med_attr, param)
        # print 'func success'

        param.feed_node_attr(ak_node)
        ak_op.set_name(med_node['ak_type'])
        ak_node.set_op(ak_op())

        # print 'name', med_node['name']
        # print 'type', type(med_node['input'])
        # print 'type', type(med_node['output'])
        [ak_node.add_in(i) for i in med_node['input']]
        [ak_node.add_out(i) for i in med_node['output']]

