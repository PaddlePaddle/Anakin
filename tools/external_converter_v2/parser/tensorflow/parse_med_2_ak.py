import numpy as np
from ..graph_io import TensorProtoIO, OpsProtoIO
from ..operations import OpsParam


def shape_2_ak_shape(shape):
    '''
    convert med shape to ak shape
    :param shape:
    :return:
    '''
    mini_shape = [i for i in shape if (i != None and i > 0)]
    return map(int, [1] * (4 - len(mini_shape)) + list(mini_shape))


def np_2_ak_tensor(np_tensor):
    '''
    conver med tensor to ak tensor
    :param np_tensor:
    :return:
    '''
    data_type_map = {
        np.dtype('float32'): 'float',
        np.dtype('int32'): 'int',
        np.dtype('bool'): 'bool'
    }

    type_str = data_type_map.get(np_tensor.dtype)
    assert type_str != None
    ak_tensor = TensorProtoIO()
    ak_tensor.set_shape(shape_2_ak_shape(np_tensor.shape))
    ak_tensor.set_data(np_tensor.flatten(), type_str)
    return ak_tensor


class MedTransAK:
    def __init__(self):
        self.input_count = 0

    def Convolution(self, med_attr, param):
        '''
        fill convlution param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        np_filters = med_attr['weights']
        param.weight_1 = np_2_ak_tensor(np_filters)
        param.filter_num = np_filters.shape[0]
        param.kernel_size = list(np_filters.shape[-2:])
        param.strides = med_attr['strides']
        param.padding = med_attr['padding']
        param.dilation_rate = med_attr['dilations']
        param.group = med_attr['group']
        param.axis = 1
        if med_attr.get('bias_weights') is not None:
            param.bias_term = True
            bias_tensor = med_attr['bias_weights']
            bias_tensor = bias_tensor.reshape(1, 1, 1, len(bias_tensor.flatten()))
            param.weight_2 = np_2_ak_tensor(bias_tensor)
        else:
            param.bias_term = False
        pass

    def Dense(self, med_attr, param):
        '''
        fill Dense param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        if med_attr.get('trans_weights', False):
            med_attr['weights'] = np.transpose(med_attr['weights'])
        param.weight_1 = np_2_ak_tensor(med_attr['weights'])
        param.axis = med_attr.get('axis', 1)
        param.out_dim = med_attr.get('out_dim', 0)

        if med_attr.get('bias_weights') is not None:
            param.bias_term = True
            param.weight_2 = np_2_ak_tensor(med_attr['bias_weights'])
            if param.out_dim == 0:
                param.out_dim = len(med_attr['bias_weights'].flatten())
        else:
            param.bias_term = False
        pass

    def Permute(self, med_attr, param):
        """
        fill Relu param in ak graph
        :param med_attr:
        :param param:
        :return:
        """
        param.dims = med_attr['dims']

    def ReLU(self, med_attr, param):
        '''
        fill Relu param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        if med_attr.get('alpha') is None:
            param.alpha = 0.0
        else:
            param.alpha = med_attr['type']

    def Activation(self, med_attr, param):
        '''
        fill Activation param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.type = med_attr['type']
        if med_attr['type'] == 'ClippedRelu':
            param.clip_relu_num = med_attr['clip_relu_num']

    def Softmax(self, med_attr, param):
        '''
        fill Softmax param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        if med_attr.get('axis') is None:
            param.axis = 3
        else:
            param.axis = med_attr['axis']
        pass

    def Concat(self, med_attr, param):
        '''
        fill Concat param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.axis = med_attr['axis']

    def Split(self, med_attr, param):
        '''
        fill Split param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.split_num = med_attr['split_num']

    def Eltwise(self, med_attr, param):
        '''
        fill Eltwise param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        assert med_attr['type'] == 'Add'
        param.type = med_attr['type']
        param.coeff = [1.0, 1.0]

    def Scale(self, med_attr, param):
        '''
        fill Scale param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.weight_1 = np_2_ak_tensor(med_attr['scale_weights'])
        if med_attr.get('bias_weights') is not None:
            param.weight_2 = np_2_ak_tensor(med_attr['bias_weights'])
            param.bias_term = True
        else:
            param.bias_term = False
        param.axis = 1
        param.num_axes = 1

    def Reshape(self, med_attr, param):
        '''
        fill Reshape param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        shape = med_attr['shape']
        if isinstance(shape, type(np.array([]))):
            shape = [int(i) for i in shape]
        param.dims = shape_2_ak_shape(shape)
        pass

    def Pooling(self, med_attr, param):
        '''
        fill Pooling param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.method = med_attr['type']
        param.pool_size = med_attr['window']
        param.strides = med_attr['strides']
        param.padding = med_attr['padding']
        if med_attr.get('global_pooling') is None:
            param.global_pooling = False
        else:
            param.global_pooling = med_attr['global_pooling']

        if med_attr.get('cmp_out_shape_floor_as_conv') is None:
            param.cmp_out_shape_floor_as_conv = False
        else:
            param.cmp_out_shape_floor_as_conv = med_attr['cmp_out_shape_floor_as_conv']

        pass

    def Pad(self, med_attr, param):
        '''
        fill Pad param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.pad_c = med_attr['pad_c']
        param.pad_h = med_attr['pad_h']
        param.pad_w = med_attr['pad_w']

    def Input(self, med_attr, param):
        '''
        fill Input param in ak graph
        :param med_attr:
        :param param:
        :return:
        '''
        param.input_shape = shape_2_ak_shape(med_attr['shape'])
        if med_attr.get('alias') is not None:
            param.alias = med_attr['alias']

    def map_med_2_ak(self, ak_node, med_node):
        '''
        entrance for trans med graph 2 ak graph
        :param ak_node:
        :param med_node:
        :return:
        '''
        type_name = med_node['ak_type']
        func = getattr(self, type_name, None)
        param = OpsParam()
        ak_op = OpsProtoIO()
        med_attr = med_node['ak_attr']
        # print('nodename = ', med_node['name'])
        func(med_attr, param)

        param.feed_node_attr(ak_node)
        ak_op.set_name(med_node['ak_type'])
        ak_node.set_op(ak_op())
        [ak_node.add_in(i['name']) for i in med_node['input']]
        [ak_node.add_out(i['name']) for i in med_node['output']]
