import numpy as np
from ..graph_io import TensorProtoIO,OpsProtoIO
from ..operations import OpsParam

def shape_2_ak_shape(shape):
    mini_shape=[i  for i in shape if (i!=None and i >0)]
    return map(int, [1] * (4 - len(mini_shape)) + list(mini_shape))

def np_2_ak_tensor(np_tensor):
    data_type_map={
        np.dtype('float32'):'float',
        np.dtype('int32'):'int',
        np.dtype('bool'):'bool'
    }

    type_str=data_type_map.get(np_tensor.dtype)
    assert type_str != None
    ak_tensor = TensorProtoIO()
    ak_tensor.set_shape(shape_2_ak_shape(np_tensor.shape))
    ak_tensor.set_data(np_tensor.flatten(),type_str)
    return ak_tensor


class MedTransAK:
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


    def Convolution(self,med_attr,param):
        np_filters=med_attr['weights']
        param.weight_1=np_2_ak_tensor(np_filters)
        param.filter_num=np_filters.shape[0]
        param.kernel_size = list(np_filters.shape[-2:])
        param.strides=med_attr['strides']
        param.padding = med_attr['padding']
        param.dilation_rate = med_attr['dilations']
        param.group = med_attr['group']
        param.axis=1
        if med_attr.get('bias_weights') is not None:
            param.bias_term = True
            bias_tensor=med_attr['bias_weights']
            bias_tensor=bias_tensor.reshape(1,1,1,len(bias_tensor.flatten()))
            param.weight_2 = np_2_ak_tensor(bias_tensor)
        else:
            param.bias_term = False


    def Dense(self, med_attr, param):
        param.weight_1 =np_2_ak_tensor(med_attr['weights'])
        param.axis=1
        if med_attr.get('bias_weights') is not None:
            param.bias_term=True
            param.weight_2=np_2_ak_tensor(med_attr['bias_weights'])
        else:
            param.bias_term=False


    def Relu(self, med_attr, param):
        if med_attr.get('alpha') is None:
            param.alpha = 0.0
        else:
            param.alpha = med_attr['type']


    def Activation(self,med_attr,param):
        param.type=med_attr['type']


    def Reshape(self,med_attr,param):
        shape=med_attr['shape']
        if isinstance(shape,type(np.array([]))):
            shape=[int(i) for i in shape]
        param.dims=shape_2_ak_shape(shape)
        pass


    def Pooling(self,med_attr,param):
        param.method=med_attr['type']
        param.pool_size=med_attr['window']
        param.strides = med_attr['strides']
        param.padding = med_attr['padding']
        if med_attr.get('global_pooling') is None:
            param.global_pooling=False
        else:
            param.global_pooling=med_attr['global_pooling']

        param.cmp_out_shape_floor_as_conv=False
        pass


    def Input(self,med_attr,param):
        param.input_shape=shape_2_ak_shape(med_attr['shape'])
        param.alias='input_'+str(self.input_count)
        self.input_count+=1


    def map_med_2_ak(self,ak_node,med_node):
        type_name=med_node['ak_type']
        func=getattr(self,type_name,None)
        param = OpsParam()
        ak_op = OpsProtoIO()
        med_attr = med_node['ak_attr']

        func(med_attr, param)

        param.feed_node_attr(ak_node)
        ak_op.set_name(med_node['ak_type'])
        ak_node.set_op(ak_op())
        [ak_node.add_in(i) for i in med_node['input']]
        [ak_node.add_out(i) for i in med_node['output']]
