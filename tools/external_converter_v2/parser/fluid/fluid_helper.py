from ..proto import *
from ..graph_io import *
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.core import VarDesc, AttrType


def union(list_a, list_b):
    '''
    '''
    return list(set(list_a).union(set(list_b)))

def difference(list_a, list_b):
    '''
    '''
    return list(set(list_a).difference(set(list_b)))


class Edge_for_fluid:

    def __init__(self, param, target, var, scale):
        '''
        '''
        self.param = param
        self.target = target
        self.var = var
        self.scale = scale


class Fluid_edger:

    def __init__(self, param=None, target=None, var=None, scale=None):
        '''
        '''
        self.edges = []
        if param is not None and target is not None:
            edge = Edge_for_fluid(param, target, var, scale)
            self.edges.append(edge)

    def __call__(self):
        '''
        '''
        return self.all_targets()

    def add(self, param, target, var=None, scale=None):
        '''
        '''
        edge = Edge_for_fluid(param, target, var, scale)
        self.edges.append(edge)

    def rm_edges_by_param(self, param):
        '''
        '''
        for edge in self.edges:
            if edge.param == param:
                edge_idx = self.edges.index(edge)
                del self.edges[edge_idx]

    def rm(self, target):
        '''
        '''
        res = -1
        for edge in self.edges:
            if target == edge.target:
                edge_idx = self.edges.index(edge)
                del self.edges[edge_idx]
                res = res + 1
        if res != 0:
            pass

    def mv(self, old_target, new_target, new_scale=None):
        '''
        '''
        res = -1
        for edge in self.edges:
            if old_target == edge.target:
                edge.target = new_target
                if new_scale is not None:
                    edge.scale = new_scale
                res = res + 1
        if res != 0:
            pass

    def reset_target_by_param(self, param, new_target):
        '''
        '''
        for edge in self.edges:
            if edge.param == param:
                edge.target = new_target

    def all_params(self):
        '''
        '''
        params = []
        for edge in self.edges:
            if edge.param not in params:
                params.append(edge.param)
        return params

    def all_targets(self):
        '''
        '''
        targets = []
        for edge in self.edges:
            targets.append(edge.target)
        return targets

    def all_scales(self):
        '''
        '''
        scales = []
        for edge in self.edges:
            scales.append(edge.scale)
        return scales

    def set_scale(self, target, scale):
        '''
        '''
        for edge in self.edges:
            if edge.target == target:
                edge.scale = scale

    def get_scale(self, target):
        '''
        '''
        for edge in self.edges:
            if edge.target == target:
                return edge.scale

    def targets(self, param):
        '''
        '''
        targets = []
        for edge in self.edges:
            if edge.param == param:
                targets.append(edge.target)
        return targets

    def target(self, param, idx = 0):
        '''
        '''
        return self.targets(param)[idx]

    def clear(self):
        '''
        '''
        targets_list = self.all_targets()
        for target in targets_list:
            self.rm(target)

    def targets_with_params(self):
        '''
        '''
        list_of_targets_and_params = []
        for edge in self.edges:
            target_and_param = [edge.target, edge.param]
            list_of_targets_and_params.append(target_and_param)
        return list_of_targets_and_params

    def vars_by_target(self, target):
        '''
        '''
        vars = []
        for edge in self.edges:
            if edge.target == target and edge.var is not None:
                vars.append(edge.var)
        return vars

    def __getitem__(self, idx):
        '''
        '''
        if idx < len(self.edges):
            return self.edges[idx]
        return None


class Fluid_helper:
    '''
    '''
    def __init__(self, scope, block):
        '''
        '''
        self.scope = scope
        self.block = block

    def args_by_input_param(self, op, param_name):
        '''
        '''
        if param_name in op.input_names:
            return op.input(param_name)
        else:
            raise NameError('ERROR: param_name %s is not exists.' % (param_name))

    def args_by_output_param(self, op, param_name):
        '''
        '''
        if param_name in op.output_names:
            return op.output(param_name)
        else:
            raise NameError('ERROR: param_name %s is not exists.' % (param_name))

    def var_by_input_param(self, op, param_name, var_idx = 0):
        '''
        '''
        var_name = self.args_by_input_param(op, param_name)[var_idx]
        var = self.block.var(var_name)
        return var

    def var_by_output_param(self, op, param_name, var_idx = 0):
        '''
        '''
        var_name = self.args_by_output_param(op, param_name)[var_idx]
        var = self.block.var(var_name)
        return var

    def var_name_by_param(self, op, param_name, var_idx = 0):
        '''
        '''
        if param_name not in op.input_names + op.output_names:
            raise NameError('ERROR: param_name %s is not exists.' % (param_name))
        elif param_name in op.input_names:
            if len(op.input(param_name)) > 0:
                var_name_unicode = op.input(param_name)[var_idx]
            else:
                raise NameError('ERROR: param %s has not var.' % (param_name))
        elif param_name in op.output_names:
            if len(op.output(param_name)) > 0:
                var_name_unicode = op.output(param_name)[var_idx]
            else:
                raise NameError('ERROR: param %s has not var.' % (param_name))
        var = self.block.var(var_name_unicode)
        var_name = var.name
        if isinstance(var_name, unicode):
            var_name = str(var_name)
        return var_name

    def var_by_param(self, op, param_name, var_idx = 0):
        '''
        '''
        var_name = self.var_name_by_param(op, param_name, var_idx)
        var = self.block.var(var_name)
        return var

    def shape_by_var_name(self, var_name, layout = 'NCHW'):
        '''
        '''
        var = self.block.var(var_name)
        long_tuple = var.shape
        long_list = list(long_tuple)
        if layout == 'NCHW':
            int_list_4d = map(int, [1] * (4 - len(long_list)) + long_list)
            return int_list_4d
        elif layout == 'UNMODIFIED':
            return long_list
        else:
            raise NameError('ERROR: layout %s is not implemented yet.' % (layout))

    def np_data_by_var_name(self, var_name):
        '''
        '''
        if hasattr(fluid.executor, '_fetch_var'):
            numpy_array = fluid.executor._fetch_var(str(var_name), self.scope, True)
        elif hasattr(fluid.executor, 'fetch_var'):
            numpy_array = fluid.executor.fetch_var(var_name, self.scope, True)
        else:
            raise NameError('ERROR: Unknown Fluid version.')
        return numpy_array

    def dtype_by_var_name(self, var_name):
        '''
        '''
        var = self.block.var(var_name)
        fluid_var_type = var.dtype
        dtype = ANAKIN_TENSOR_DTYPE[fluid_var_type]
        return dtype

    def is_persistable_param(self, op, param_name, var_idx = 0):
        '''
        '''
        var = self.var_by_param(op, param_name, var_idx)
        is_persistable_var = var.persistable
        return is_persistable_var

    def var_shape_by_param(self, transpose, op, param_name, var_idx = 0, layout = 'NCHW'):
        '''
        '''
        if transpose is True:
            raise NameError('ERROR: var_shape transpose is not implemented yet.')
        else:
            var_name = self.var_name_by_param(op, param_name, var_idx)
            shape = self.shape_by_var_name(var_name, layout)
            return shape

    def data_with_shape_by_param(self,
                                 op,
                                 param_name,
                                 transpose = False,
                                 axes = None,
                                 var_idx = 0,
                                 is_flat_list = True,
                                 layout = 'NCHW'):
        '''
        '''
        np.set_printoptions(threshold=np.inf, suppress=True)

        var_name = self.var_name_by_param(op, param_name, var_idx)
        np_array = self.np_data_by_var_name(var_name)
        if transpose is True:
            np_array = np.transpose(np_array, axes)
        np_shape = np.shape(np_array)
        if layout == 'NCHW':
            np_shape = map(int, [1] * (4 - len(np_shape)) + list(np_shape))
        if is_flat_list is True:
            flat_list = np_array.flatten().tolist()
            return [flat_list, np_shape]
        else:
            return [np_array, np_shape]

    def np_param(self,
                 op,
                 param_name,
                 transpose = False,
                 axes = None,
                 var_idx = 0):
        '''
        '''
        [data, np_shape] = self.data_with_shape_by_param(op, param_name, transpose, \
            axes, var_idx, False)
        return data

    def dtype_by_param(self, op, param_name, var_idx = 0):
        '''
        '''
        var_name = self.var_name_by_param(op, param_name, var_idx)
        dtype = self.dtype_by_var_name(var_name)
        return dtype

    def is_list_type(self, op, attr_name):
        '''
        '''
        if op.has_attr(attr_name):
            fluid_attr_type = op.attr_type(attr_name)
            if fluid_attr_type in ANAKIN_ATTR_IS_LIST.keys():
                return ANAKIN_ATTR_IS_LIST[fluid_attr_type]
            else:
                return False # AttrType.LONG
        else:
            raise NameError('ERROR: attr_name %s is not exists.' % (attr_name))

    def dtype_of_attr(self, op, attr_name):
        '''
        '''
        if op.has_attr(attr_name):
            fluid_attr_type = op.attr_type(attr_name)
            if fluid_attr_type in ANAKIN_ATTR_DTYPE.keys():
                return ANAKIN_ATTR_DTYPE[fluid_attr_type]
            else:
                return INT32 # AttrType.LONG
        else:
            raise NameError('ERROR: attr_name %s is not exists.' % (attr_name))

    def attr_data_required(self, op, attr_name):
        '''
        '''
        data = op.attr(attr_name)
        is_list = self.is_list_type(op, attr_name)
        dtype = self.dtype_of_attr(op, attr_name)
        if dtype not in [INT32, FLOAT, STR]:
            return data
        elif dtype == INT32:
            return map(int, data) if is_list else int(data)
        elif dtype == FLOAT:
            return map(float, data) if is_list else float(data)
        elif dtype == STR:
            return bytes(data)

    def attr_data(self, op, attr_name, default_value = 0, type = None):
        '''
        '''
        if op.has_attr(attr_name):
            return self.attr_data_required(op, attr_name)
        else:
            #raise NameError('ERROR: attr_name %s is not exists.' % (attr_name))
            return default_value

    def param_tensor_sh(self,
                        op,
                        param_name,
                        transpose = False,
                        axes = None,
                        reshape = None,
                        var_idx = 0,
                        layout = 'NCHW'):
        '''
        '''
        tensor = TensorProtoIO()
        [np_data, shape] = self.data_with_shape_by_param(op, param_name, transpose, \
            axes, var_idx, False, layout)
        dtype = self.dtype_by_param(op, param_name, var_idx)
        tensor.set_data_type(dtype)
        if dtype is INT8:
            tensor.set_data(np_data.flatten().tobytes(), ANAKIN_TENSOR_DTYPESTR[dtype])
        elif dtype in ANAKIN_TENSOR_DTYPESTR.keys():
            tensor.set_data(np_data.flatten().tolist(), ANAKIN_TENSOR_DTYPESTR[dtype])
            #pass #debug
        else:
            raise NameError('ERROR: Unknown data type (%s)' % (dtype))
        if reshape is not None:
            tensor.set_shape(reshape)
        else:
            tensor.set_shape(shape)
        return [tensor, shape]

    def param_tensor(self,
                     op,
                     param_name,
                     transpose = False,
                     axes = None,
                     reshape = None,
                     var_idx = 0,
                     layout = 'NCHW'):
        '''
        '''
        [tensor, shape] = self.param_tensor_sh(op, param_name, transpose, axes, \
            reshape, var_idx, layout)
        return tensor

    def create_tensor(self, data_list, data_shape, dtype, scale=None):
        '''
        '''
        tensor = TensorProtoIO()
        tensor.set_data_type(dtype)
        tensor.set_data(data_list, ANAKIN_TENSOR_DTYPESTR[dtype])
        tensor.set_shape(data_shape)
        if scale is not None:
            tensor.set_scale(scale)
        return tensor

    def gru_tensor_convert(self, origin_h2h, origin_i2h, origin_b, offset=[2, 1, 0]):
        '''
        '''
        hidden_size = int(origin_b.size // 3)
        word_size = int(origin_i2h.size // hidden_size // 3)
        tar_h2h = np.array(origin_h2h.flatten().tolist()[2 * hidden_size * hidden_size:]\
            + np.array(origin_h2h.flatten().tolist()[: 2 * hidden_size * hidden_size])\
            .reshape(hidden_size, 2, hidden_size)[:, [1, 0], :].flatten().tolist())\
        .reshape(1, 1, hidden_size, 3 * hidden_size)
        tar_i2h = origin_i2h.reshape(word_size, 3, hidden_size)[:, offset, :]\
        .reshape(1, 1, word_size, 3 * hidden_size)
        tar_b = origin_b.reshape(3, hidden_size)[offset, :].reshape(1, 1, 1, 3 * hidden_size)
        tar_i2h_h2h = np.concatenate([tar_i2h.flatten(), tar_h2h.flatten()])\
        .reshape(1, 1, 1, 3 * hidden_size * hidden_size + 3 * word_size * hidden_size)
        return tar_i2h_h2h, tar_b

    def lstm_fc_tensor_merge_convert(self,
                                     origin_hidden_size,
                                     origin_lstm_w,
                                     origin_lstm_b,
                                     origin_fc_w,
                                     origin_fc_b):
        '''
        '''
        layer_size = int(origin_hidden_size // 4)
        input_size = int(origin_fc_w.size // origin_hidden_size)
        lstm_bias_num = int(origin_lstm_b.size // layer_size)
        tar_w = np.vstack((np.hstack((origin_fc_w[:, 1 * layer_size: 2 * layer_size],
                                      origin_fc_w[:, 2 * layer_size: 3 * layer_size],
                                      origin_fc_w[:,: 1 * layer_size],
                                      origin_fc_w[:, 3 * layer_size:])),
                           np.hstack((origin_lstm_w[:, 1 * layer_size: 2 * layer_size],
                                      origin_lstm_w[:, 2 * layer_size: 3 * layer_size],
                                      origin_lstm_w[:, : 1 * layer_size],
                                      origin_lstm_w[:, 3 * layer_size:]))))

        if origin_fc_b is not None:
            split_fc_bc = origin_fc_b.flatten()[: 1 * layer_size]
            split_fc_bi = origin_fc_b.flatten()[1 * layer_size : 2 * layer_size]
            split_fc_bf = origin_fc_b.flatten()[2 * layer_size : 3 * layer_size]
            split_fc_bo = origin_fc_b.flatten()[3 * layer_size : 4 * layer_size]
        else:
            split_fc_bc = np.zeros(layer_size)
            split_fc_bi = np.zeros(layer_size)
            split_fc_bf = np.zeros(layer_size)
            split_fc_bo = np.zeros(layer_size)

        split_lstm_bc = origin_lstm_b.flatten()[: 1 * layer_size]
        split_lstm_bi = origin_lstm_b.flatten()[1 * layer_size: 2 * layer_size]
        split_lstm_bf = origin_lstm_b.flatten()[2 * layer_size: 3 * layer_size]
        split_lstm_bo = origin_lstm_b.flatten()[3 * layer_size: 4 * layer_size]
        split_lstm_bc = np.add(split_lstm_bc, split_fc_bc)
        split_lstm_bi = np.add(split_lstm_bi, split_fc_bi)
        split_lstm_bf = np.add(split_lstm_bf, split_fc_bf)
        split_lstm_bo = np.add(split_lstm_bo, split_fc_bo)

        if lstm_bias_num == 4:
            tar_b = np.array(split_lstm_bi.flatten().tolist()
                             + split_lstm_bf.flatten().tolist()
                             + split_lstm_bc.flatten().tolist()
                             + split_lstm_bo.flatten().tolist())
        else:
            split_lstm_wic = origin_lstm_b.flatten()[4 * layer_size : 5 * layer_size]
            split_lstm_wfc = origin_lstm_b.flatten()[5 * layer_size : 6 * layer_size]
            split_lstm_woc = origin_lstm_b.flatten()[6 * layer_size :]
            tar_b = np.array(split_lstm_bi.flatten().tolist()
                             + split_lstm_bf.flatten().tolist()
                             + split_lstm_bc.flatten().tolist()
                             + split_lstm_bo.flatten().tolist()
                             + split_lstm_wic.flatten().tolist()
                             + split_lstm_wfc.flatten().tolist()
                             + split_lstm_woc.flatten().tolist())
        return tar_w.reshape(input_size + layer_size, 4 * layer_size, 1, 1),\
               tar_b.reshape(1, origin_lstm_b.size, 1, 1)


class Fluid_comparator:
    """
    """
    def __init__(self, helper):
        """
        """
        self.helper = helper
        self.only_list = ['feed', 'fetch']

    def compare_by_param(self, op_a, op_b, param):
        """
        """
        is_weight_a = self.helper.is_persistable_param(op_a, param)
        is_weight_b = self.helper.is_persistable_param(op_b, param)
        if is_weight_a and is_weight_b:
            np_a = self.helper.np_param(op_a, param)
            np_b = self.helper.np_param(op_b, param)
            if (np_a == np_b).all() == True:
                return True
            else:
                return False
        elif is_weight_a is is_weight_b:
            return True
        else:
            return False

    def have_same_weights(self, op_a, op_b):
        """
        """
        is_same = True
        if op_a.input_names == op_b.input_names:
            params = op_a.input_names
            for param in params:
                if self.compare_by_param(op_a, op_b, param) is False:
                    is_same = False
            return is_same
        else:
            return False

    def compare_by_attr(self, op_a, op_b, attr_name):
        """
        """
        data_a = self.helper.attr_data(op_a, attr_name)
        data_b = self.helper.attr_data(op_b, attr_name)
        return data_a == data_b

    def have_same_attrs(self, op_a, op_b):
        """
        """
        is_same = True
        if op_a.attr_names == op_b.attr_names:
            attrs = op_a.attr_names
            for attr in attrs:
                if self.compare_by_attr(op_a, op_b, attr) is False:
                    is_same = False
            return is_same
        else:
            return False

    def brothers(self, op_list):
        """
        """
        is_same = True
        if len(op_list) > 1:
            idx = 0
            for op_b in op_list[1:]:
                if op_b.type not in self.only_list:
                    idx = op_list.index(op_b)
                    op_a = op_list[idx - 1]
                    if op_a.type not in self.only_list:
                        same_weights = self.have_same_weights(op_a, op_b)
                        same_attrs = self.have_same_attrs(op_a, op_b)
                        if (same_weights and same_attrs) is False:
                            is_same = False
                    else:
                        raise NameError('ERROR: %s is in only_list.' % (op_a.type))
                else:
                    raise NameError('ERROR: %s is in only_list.' % (op_b.type))
            return is_same
        else:
            raise NameError('ERROR: Members of op_list must be greater than 2.')


ANAKIN_TENSOR_DTYPE = {
    VarDesc.VarType.INT8: INT8,
    VarDesc.VarType.BOOL: BOOLEN,
    VarDesc.VarType.INT32: INT32,
    VarDesc.VarType.FP16: FLOAT16,
    VarDesc.VarType.FP32: FLOAT,
    VarDesc.VarType.FP64: DOUBLE,
}

ANAKIN_TENSOR_DTYPESTR = {
    STR: "string",
    INT8: "int8",
    INT32: "int32",
    FLOAT: "float",
    BOOLEN: "bool"
}

ANAKIN_ATTR_DTYPE = {
    AttrType.INT: INT32,
    AttrType.INTS: INT32,
    AttrType.FLOAT: FLOAT,
    AttrType.FLOATS: FLOAT,
    AttrType.STRING: STR,
    AttrType.STRINGS: STR,
    AttrType.BOOL: BOOLEN,
    AttrType.BOOLS: BOOLEN,
}

ANAKIN_ATTR_IS_LIST = {
    AttrType.INT: False,
    AttrType.INTS: True,
    AttrType.FLOAT: False,
    AttrType.FLOATS: True,
    AttrType.STRING: False,
    AttrType.STRINGS: True,
    AttrType.BOOL: False,
    AttrType.BOOLS: True,
}

APPEND_BIAS_OP_TYPE = [
    'FC',
    'mul',
    'sequence_conv',
    'conv2d',
    'conv2d_transpose',
    'depthwise_conv2d',
    'elementwise_mul',
]

APPEND_ACT_OP_TYPE = [
    'FC',
    'mul',
    'sequence_conv',
    'conv2d',
    'conv2d_transpose',
    'batch_norm',
    'layer_norm',
    'row_conv',
    'reshape',
]

FLUID_QUANTIZE_LAYERS = [
    'fake_quantize_abs_max',
    'fake_quantize_range_abs_max',
    'quantize',
]

FLUID_DEQUANTIZE_LAYERS = [
    'fake_dequantize_max_abs',
    'fake_dequantize_range_max_abs',
    'dequantize',
]

FLUID_SCALE_WEIGHT_OP = [
    'conv2d',
    'depthwise_conv2d',
]

