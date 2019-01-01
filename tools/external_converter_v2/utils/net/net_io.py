#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append("../../")

from parser.proto import net_pb2
from parser.graph_io import GraphProtoIO
from google.protobuf import text_format


class FuncProtoIO(object):
    """
    Func io class of FuncProto.
    """

    def __init__(self, proto=None):
        """
        """
        self.func_proto = None
        if proto is None:
            self.func_proto = net_pb2.FuncProto()
        else:
            self.func_proto = proto

    def get_name(self):
        return self.func_proto.name

    def set_name(self, name):
        self.func_proto.name = name

    def get_type(self):
        return self.func_proto.type

    def set_type(self, type_value):
        self.func_proto.type = type_value

    def get_node_io(self):
        node_io = NodeProtoIO(self.func_proto.node_info)
        return node_io

    def reset_node_io(self, node_io):
        node_proto = node_io()
        self.func_proto.node_info.CopyFrom(node_proto)

    def __call__(self):
        return self.func_proto


class NetProtoIO(object):
    """
    Net io class of NetProto.
    """

    def __init__(self, proto=None):
        """
        """
        self.net_proto = None
        if proto is None:
            self.net_proto = net_pb2.NetProto()
        else:
            self.net_proto = proto

    def graph_io(self):
        graph_io = GraphProtoIO(self.net_proto.graph)
        return graph_io

    def clear_graph(self):
        self.net_proto.graph.Clear()

    def get_name(self):
        return self.net_proto.name

    def set_name(self, net_name):
        self.net_proto.name = net_name

    def add_func(self, func=net_pb2.FuncProto()):
        self.net_proto.funcs.extend([func])

    def func_io_list(self):
        func_io_list = list()
        for func in self.net_proto.funcs:
            func_io = FuncProtoIO(func)
            func_io_list.append(func_io)
        return func_io_list

    def save(self, file_path, use_txt=True, use_net_name=True):
        """
        """
        if use_net_name is True:
            assert self.net_proto.name is not None
            file_path = os.path.join(file_path, self.net_proto.name)
        with open(file_path, "wb") as f:
            if use_txt is True:
                f.write(text_format.MessageToString(self.net_proto))
            else:
                f.write(self.net_proto.SerializeToString())
        f.close()

    def parse_from_string(self, file_path):
        """
        parser from optimized graph model
        """
        with open(file_path, "rb") as f:
            contents = f.read()
            self.net_proto.ParseFromString(contents)

    def merge_from_string(self, file_path):
        """
        parser from optimized graph model
        """
        with open(file_path, "rb") as f:
            contents = f.read()
            self.net_proto.MergeFromString(contents)

    def __call__(self):
        return self.net_proto
