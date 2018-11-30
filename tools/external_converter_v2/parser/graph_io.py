#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import numpy as np
from google.protobuf import text_format
from utils import *
from proto import *


class NodeAttrWrapper(object):
    """
    """

    def __init__(self):
        self.value_data = valueType()

    def __call__(self, data, data_type_str):
        """
        """
        if data_type_str == type(""):  # type string
            self.value_data.s = data
            self.value_data.type = STR
        elif data_type_str == type(int()):  # type int
            self.value_data.i = data
            self.value_data.type = INT32
        elif data_type_str == type(float()):  # type float
            self.value_data.f = data
            self.value_data.type = FLOAT
        elif data_type_str == type(bool()):  # type bool
            self.value_data.b = data
            self.value_data.type = BOOLEN
        elif data_type_str == type(TensorProtoIO()):  # type tensor
            self.value_data.tensor.CopyFrom(data())
            self.value_data.type = TENSOR
        elif data_type_str == type(unicode()):  # not used
            return self.value_data
        elif data_type_str == type(list()):  # type shape
            self.value_data.type = CACHE_LIST
            if len(data):  # in case of error(empty data list): index out of range
                if type(data[0]) == type(float()):
                    self.value_data.cache_list.f[:] = data
                    self.value_data.cache_list.type = FLOAT
                    self.value_data.cache_list.size = len(data)
                elif type(data[0]) == type(bool()):
                    self.value_data.cache_list.b[:] = data
                    self.value_data.cache_list.type = BOOLEN
                    self.value_data.cache_list.size = len(data)
                elif type(data[0]) == type(int()) or type(data[0]) == type(long()):
                    self.value_data.cache_list.i[:] = data
                    self.value_data.cache_list.type = INT32
                    self.value_data.cache_list.size = len(data)
                elif type(data[0]) == type(""):
                    self.value_data.cache_list.s[:] = data
                    self.value_data.cache_list.type = STR
                    self.value_data.cache_list.size = len(data)
                elif type(data[0]) == type(data):  # Recursive Structures of list..[list...] (deep num is only 2)
                    self.value_data.cache_list.type = CACHE_LIST
                    self.value_data.cache_list.size = len(data)
                    for idx, list_one in enumerate(data):
                        if type(list_one[0]) == type(int()) or type(list_one[0]) == type(long()):
                            data_cache = CacheDate()
                            data_cache.i[:] = list_one
                            data_cache.type = INT32
                            data_cache.size = len(list_one)
                            self.value_data.cache_list.l.extend([data_cache])
                        else:
                            raise NameError(
                                'ERROR: UnSupport Recursive list data type(%s) in list '
                                % (str(type(list_one[0])))
                            )
                else:
                    raise NameError('ERROR: UnSupport data type(%s) in list ' % (str(type(data[0]))))
            else:
                self.value_data.cache_list.f[:] = data
                self.value_data.cache_list.type = FLOAT
                self.value_data.cache_list.size = len(data)
        else:
            raise NameError('ERROR: Unknown data type (%s) in message valueType' % (data_type_str))
        return self.value_data


class TensorProtoIO(object):
    """
    """

    def __init__(self):
        """
        """
        self.tensor_proto = TensorProto()
    
    def set_shared(self, is_shared):
        self.tensor_proto.shared = is_shared
    
    def set_shared_from(self, shared_node_name):
        # current tensor is shared from the node shared_node_name if it needs.
        self.tensor_proto.share_from = shared_node_name
    
    def set_data_type(self, data_type):
        self.tensor_proto.data.type = data_type

    def get_shape(self):
        return self.tensor_proto.shape.dim.value

    def set_shape(self, shape_list):
        """
        Shape list equal to python list
        """
        self.tensor_proto.shape.dim.value[:] = shape_list
        self.tensor_proto.shape.dim.size = len(shape_list)

    def get_data(self):
        """
        """
        if self.tensor_proto.data.type == STR:
            return self.tensor_proto.data.s
        elif self.tensor_proto.data.type == INT32:
            return self.tensor_proto.data.i
        elif self.tensor_proto.data.type == FLOAT:
            return self.tensor_proto.data.f
        elif self.tensor_proto.data.type == BOOLEN:
            return self.tensor_proto.data.b
        else:
            raise NameError('ERROR: Unknown data type in message CacheDate')

    def set_data(self, data_list, data_type):
        """
        """
        if data_type == "string":
            self.tensor_proto.data.s[:] = data_list
            self.tensor_proto.data.type = STR
        elif data_type == "int32":
            self.tensor_proto.data.i[:] = data_list
            self.tensor_proto.data.type = INT32
        elif data_type == "int8":
            assert type(data_list) is str
            self.tensor_proto.data.c = data_list
            self.tensor_proto.data.type = INT8
        elif data_type == "float":
            self.tensor_proto.data.f[:] = data_list
            self.tensor_proto.data.type = FLOAT
        elif data_type == "bool":
            self.tensor_proto.data.b[:] = data_list
            self.tensor_proto.data.type = BOOLEN
        else:
            raise NameError('ERROR: Unknown data type (%s) in message CacheDate' % (data_type))
        self.tensor_proto.data.size = len(data_list)

    def set_scale(self, data_list, data_type):
        """
        """
        if data_type == "float":
            self.tensor_proto.scale.f[:] = data_list
            self.tensor_proto.scale.type = FLOAT
        else:
            raise NameError('ERROR: Unknown data type (%s) in message CacheDate' % (data_type))
        self.tensor_proto.scale.size = len(data_list)

    def __call__(self):
        return self.tensor_proto


class OpsProtoIO(object):
    """
    """

    def __init__(self):
        """
        """
        self.op_proto = OpsProto()

    def set_name(self, op_name):
        self.op_proto.name = op_name

    def set_commutative(self, bool_value):
        self.op_proto.is_commutative = bool_value

    def set_in_num(self, num_elem):
        self.op_proto.in_num = num_elem

    def set_out_num(self, num_elem):
        self.op_proto.out_num = num_elem

    def set_desc(self, description):
        self.op_proto.description = description

    def __call__(self):
        return self.op_proto

class NodeProtoIO(object):
    """
    Node io class of NodeProto
    """

    def __init__(self):
        """
        """
        self.node_proto = NodeProto()
        self.attr_warpper = NodeAttrWrapper()

    def set_name(self, node_name):
        self.node_proto.name = node_name

    def add_in(self, node_name):
        self.node_proto.ins.append(node_name)

    def add_out(self, node_name):
        self.node_proto.outs.append(node_name)

    def set_op(self, operator=OpsProto()):
        self.node_proto.Op.CopyFrom(operator)

    def set_bit_type(self, bit_type):
        """
        Bit width setting.
        """
        self.node_proto.bit_type = bit_type

    def add_attr(self, value_name, data, data_type_str):
        """
        set tensor data:
                value_name : var name
                data       : real data
                data_type_str : data type desc ("string"
                                                                                "int"
                                                                                "float"
                                                                                "bool"
                                                                                "tensor"
                                                                                "shape"
                                                                                "list_value")
        """

        self.node_proto.attr[value_name].CopyFrom(self.attr_warpper(data, data_type_str))

    def __call__(self):
        return self.node_proto


class GraphProtoIO(object):
    """
    Graph io class of GraphProto.
    """

    def __init__(self):
        """
        """
        self.graph_proto = GraphProto()

    def serialization(self, file_path):
        """
        Serialize to disk.
        """
        # self._get_graph_proto();
        with open(file_path, "wb") as f:
            f.write(self.graph_proto.SerializeToString())
        f.close()

    def parse_from_string(self, file_path):
        """
        parser from optimized graph model
        """
        with open(file_path, "rb") as f:
            contents = f.read()
            self.graph_proto.ParseFromString(contents)

    def set_name(self, graph_name):
        self.graph_proto.name = graph_name

    def add_node(self, node=NodeProtoIO()):
        self.graph_proto.nodes.extend([node])

    def rm_node(self, node):
        if node in self.graph_proto.nodes:
            index = -1
            for idx, tmp_node in enumerate(self.graph_proto.nodes):
                if tmp_node == node:
                    index = idx
                    break
            if index >= 0:
                del self.graph_proto.nodes[index]
        else:
            raise NameError('ERROR: (%s) node not exist.' % (node))

    def find_node_proto(self, node_name):
        for node in self.graph_proto.nodes:
            if node.name == node_name:
                return node

    def get_edge_nexts(self, node_name, with_info=False):
        """
        get edge's next node_name
        """
        edges_out = self.graph_proto.edges_out
        nexts = list()
        if node_name in edges_out:
            if with_info is False:
                for target in edges_out[node_name].target:
                    nexts.append(target.node)
            else:
                nexts = edges_out[node_name].target[:]
        return nexts

    def rm_edge(self, node_name_0, node_name_1):
        """
        remove edge is directive from node_name_0 to node_name_1
        """
        if node_name_0 in self.graph_proto.edges_out:
            index = -1
            for idx, target in enumerate(self.graph_proto.edges_out[node_name_0].target):
                if target.node == node_name_1:
                    index = idx
                    break
            if index >= 0:
                # print "suc in " + node_name_0 + " -> " + node_name_1 + "  idx: "  + str(index)
                del self.graph_proto.edges_out[node_name_0].target[index]
        if node_name_1 in self.graph_proto.edges_in:
            index = -1
            for idx, target in enumerate(self.graph_proto.edges_in[node_name_1].target):
                if target.node == node_name_0:
                    index = idx
                    break
            if index >= 0:
                # print "suc in " + node_name_0 + " -> " + node_name_1 +  " idx: " + str(index)
                del self.graph_proto.edges_in[node_name_1].target[index]

    def add_in_edge(self, node_name_0, node_name_1, scale=None):
        """
        add_in_edge is directive from node_name_0 to node_name_1
        """
        edges_in = self.graph_proto.edges_in
        nexts = list()
        for target in edges_in[node_name_1].target:
            nexts.append(target.node)
        if node_name_0 not in nexts:
            target = edges_in[node_name_1].target.add()
            if scale is not None:
                target.scale.append(scale)
            target.node = node_name_0

    def add_out_edge(self, node_name_0, node_name_1, scale=None):
        """
        add_out_edge is directive from node_name_0 to node_name_1
        """
        edges_out = self.graph_proto.edges_out
        nexts = list()
        for target in edges_out[node_name_0].target:
            nexts.append(target.node)
        if node_name_1 not in nexts:
            target = edges_out[node_name_0].target.add()
            if scale is not None:
                target.scale.append(scale)
            target.node = node_name_1

    def add_in(self, node_name):
        self.graph_proto.ins.append(node_name)

    def rm_in(self, node_name):
        graph_ins = list(self.graph_proto.ins)
        for in_name in graph_ins:
            if node_name == in_name:
                idx = graph_ins.index(in_name)
                del graph_ins[idx]
        self.graph_proto.ins[:] = graph_ins

    def ins(self):
        return list(self.graph_proto.ins)

    def outs(self):
        return list(self.graph_proto.outs)

    def add_out_fluid(self, output_node_name, in_node_name):
        """
        add output node for graph
        """
        nodeIO = NodeProtoIO()
        nodeIO.set_name(output_node_name)
        nodeIO.add_in(in_node_name)
        opIO = OpsProtoIO()
        opIO.set_name("Output")
        nodeIO.set_op(opIO())
        self.add_node(nodeIO())
        self.graph_proto.outs.append(output_node_name)

    def add_out(self, output_node_name, in_node_name):
        """
        add output node for graph
        """
        nodeIO = NodeProtoIO()
        nodeIO.set_name(output_node_name)
        nodeIO.add_in(in_node_name)
        opIO = OpsProtoIO()
        opIO.set_name("Output")
        nodeIO.set_op(opIO())
        self.add_out_edge(in_node_name, output_node_name)
        self.add_in_edge(in_node_name, output_node_name)
        self.add_node(nodeIO())
        self.graph_proto.outs.append(output_node_name)

    def rm_out(self, node_name):
        graph_outs = list(self.graph_proto.outs)
        for out_name in graph_outs:
            if node_name == out_name:
                idx = graph_outs.index(out_name)
                del graph_outs[idx]
        self.graph_proto.outs[:] = graph_outs

    def format_edge_from_nodes(self):
        """
        format edge from nodes with input and output list
        :return:
        """
        in_set = set()
        out_set = set()
        for node in self.graph_proto.nodes:
            name = node.name
            for node_name in node.ins:
                self.add_in_edge(node_name, name)
                in_set.add((node_name, name))
            for node_name in node.outs:
                self.add_out_edge(name, node_name)
                out_set.add((name, node_name))
        ab_set = in_set - out_set
        ba_set = out_set - in_set
        print(ab_set)
        print('------')
        print(ba_set)
        assert len(ab_set) == 0 and len(ba_set) == 0, 'in edge must equal with out edge'


    def __call__(self):
        return self.graph_proto

