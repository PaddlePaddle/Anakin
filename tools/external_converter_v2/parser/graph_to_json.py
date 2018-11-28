#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import random
from utils import *
from proto import *
#from kill_caffe import *
from logger import *
from graph_io import GraphProtoIO


class CreateJson(object):
    """
    create an json str from dict
    """
    def __init__(self, **kwargs):
        self.__dict__ = {}
        self.__dict__ = kwargs

    def __call__(self):
        return self.__dict__


class GraphToJson(object):
    """
    convert GraphProto to json(dict type) which can be used by cytoscape elements
    """
    def __init__(self, graph_io=GraphProtoIO()):
        self.graph_proto = graph_io()
        # decide layout
        #self.get_layout_coordinate()

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

    def get_layout_coordinate(self):
        """
        get layout coordinate of node in graph board
        """
        # map node_name to its coordinate list(x, y)
        self.map_node_to_coordinate = dict()
        start_cord_x = 0
        horizon_step = 500
        vertical_step = 200
        # assign start nodes
        start_nodes = self.graph_proto.ins
        for node in start_nodes:
            self.map_node_to_coordinate[node] = [start_cord_x, 0]
            start_cord_x = start_cord_x + horizon_step
        # assign middle nodes
        nodes = self.graph_proto.nodes
        while len(self.map_node_to_coordinate.keys()) < len(nodes):
            for node_proto in nodes:
                if node_proto.name in self.map_node_to_coordinate.keys():
                    x = self.map_node_to_coordinate[node_proto.name][0]
                    y = self.map_node_to_coordinate[node_proto.name][1]
                    inc_step = 0
                    for next_node_name in self.get_edge_nexts(node_proto.name):
                        self.map_node_to_coordinate[next_node_name] = [0, 0]
                        self.map_node_to_coordinate[next_node_name][0] = x + inc_step
                        inc_step = inc_step + horizon_step
                        self.map_node_to_coordinate[next_node_name][1] = y + vertical_step
                else: 
                    break

    def create_nodes(self):
        """
        create graph nodes
        """
        nodes = []
        exec_count = 0
        for node_proto in self.graph_proto.nodes:
            inner_data = CreateJson(id=node_proto.name,
                                                            name=node_proto.Op.name,
                                                            exec_count=exec_count,
                                                            lane=node_proto.lane, # addition for optimization
                                                            need_wait=node_proto.need_wait) # addition for optimization
            exec_count = exec_count + 1
            node = CreateJson(data=inner_data())
            nodes.append(node())
        return nodes

    def create_edges(self):
        """
        create graph edges
        """
        memory_bank_count = 1
        # map : edge name --> hex color of html
        self.edge_color_map = dict()
        # map : edge name --> memory bank id
        self.edge_memory_bank_map = dict()
        r = lambda: random.randint(0, 255)
        new_color = lambda: ("#%02X%02X%02X" % (r(), r(), r()))
        for node_proto in self.graph_proto.nodes:
            if node_proto.name in self.graph_proto.edges_out:
                for node_name in self.get_edge_nexts(node_proto.name):
                    edge_name = node_proto.name + '_' + node_name
                    if edge_name in self.graph_proto.edges_info:
                        tensor_proto = self.graph_proto.edges_info[edge_name]
                        shared = tensor_proto.shared
                        tensor_name = tensor_proto.name
                        if not shared:
                            self.edge_color_map[tensor_name] = new_color()
                            self.edge_memory_bank_map[tensor_name] = memory_bank_count
                            memory_bank_count = memory_bank_count + 1
        edges = []
        for node_proto in self.graph_proto.nodes:
            if node_proto.name in self.graph_proto.edges_out:
                for node_name in self.get_edge_nexts(node_proto.name):
                    edge_name = node_proto.name + '_' + node_name
                    tensor_name = ""
                    shared = ""
                    share_from = ""
                    if edge_name in self.graph_proto.edges_info:
                        tensor_proto = self.graph_proto.edges_info[edge_name]
                        tensor_name = tensor_proto.name
                        shared = tensor_proto.shared
                        share_from = tensor_proto.share_from
                        if shared:
                            self.edge_color_map[tensor_name] = self.edge_color_map[share_from]
                            self.edge_memory_bank_map[tensor_name] = self.edge_memory_bank_map[share_from]
                    if tensor_name == "":
                        inner_data = CreateJson(source=node_proto.name,
                                                                        target=node_name)
                    else:
                        inner_data = CreateJson(source=node_proto.name,
                                                                        target=node_name,
                                                                        edge_name=tensor_name,
                                                                        shared=shared,
                                                                        share_from=share_from,
                                                                        edge_color=self.edge_color_map[tensor_name],
                                                                        memory_id=self.edge_memory_bank_map[tensor_name])
                    edge = CreateJson(data=inner_data())
                    edges.append(edge())
        return edges

    def create_attr(self):
        """
        create graph attrs
        """
        attrs = []
        for node_proto in self.graph_proto.nodes:
            key_id = node_proto.name
            node_attrs = []
            for attr_name in node_proto.attr.keys():
                value_data = node_proto.attr[attr_name]
                name = attr_name
                type_str = ""
                value = ""
                if value_data.type == STR:
                    type_str = "string"
                    value = value_data.s
                elif value_data.type == INT32:
                    type_str = "int"
                    value = value_data.i
                elif value_data.type == FLOAT:
                    type_str = "float"
                    value = value_data.f
                elif value_data.type == BOOLEN:
                    type_str = "bool"
                    value = "true" if value_data.b else "false"
                elif value_data.type == TENSOR:
                    type_str = "tensor"
                    value = list(value_data.tensor.shape.dim.value)
                elif value_data.type == CACHE_LIST:
                    type_str = "list"
                    if value_data.cache_list.type == STR:
                        value = value_data.cache_list.s
                    elif value_data.cache_list.type == INT32:
                        value = value_data.cache_list.i
                    elif value_data.cache_list.type == FLOAT:
                        value = value_data.cache_list.f
                    elif value_data.cache_list.type == BOOLEN:
                        value = value_data.cache_list.b
                    elif value_data.cache_list.type == CACHE_LIST:
                        value = []
                        for list_tmp in value_data.cache_list.l:
                            value.append(list_tmp.i)
                    else:
                        raise NameError('ERROR: UnSupport Recursive list data type(%s) in list ' % (str(value_data.cache_list.type)))
                else:
                    raise NameError('ERROR: Unknown data type (%s) in message valueType' % (str(value_data.type)))
                target_attr = CreateJson(id=name, 
                                                                 type=type_str,
                                                                 value=str(value))
                node_attrs.append(target_attr())
            # Quantitative information
            name = 'bit_mode'
            type_str = 'type'
            if node_proto.bit_type == FLOAT:
                value = 'FLOAT32'
            elif node_proto.bit_type == INT8:
                value = 'INT8'
            elif node_proto.bit_type == STR:
                value = 'UNKNOWN'
            else:
                raise NameError('ERROR: Unknown data type (%d) in message valueType' \
                 % (node_proto.bit_type))
            target_attr = CreateJson(id=name, 
                                                             type=type_str,
                                                             value=str(value))
            node_attrs.append(target_attr())
            node_map = CreateJson(key_name=key_id,
                                                      key_attrs=node_attrs)
            attrs.append(node_map())
        return attrs

    def create_elements(self):
        """
        create elements of graph for cytoscape
        """
        elements = CreateJson(nodes=self.create_nodes(),
                                                  edges=self.create_edges())
        return elements()

    def create_mem_info(self):
        """
        create memory optimization information
        """
        temp_mem_used = self.graph_proto.summary.temp_mem_used
        system_mem_used = self.graph_proto.summary.system_mem_used
        model_mem_used = self.graph_proto.summary.model_mem_used
        sum_mem = self.graph_proto.summary.temp_mem_used + \
				self.graph_proto.summary.system_mem_used + \
				self.graph_proto.summary.model_mem_used
        mem_info =  CreateJson(temp_mem=temp_mem_used, 
							   system_mem=system_mem_used, 
							   model_mem=model_mem_used,
							   total_mem=sum_mem)
        return mem_info()

    def __call__(self):
        return self.create_elements(), self.create_attr(), self.create_mem_info()
