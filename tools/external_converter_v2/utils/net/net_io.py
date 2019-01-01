#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")

from parser.proto import net_pb2
from parser.graph_io import GraphProtoIO

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

    def func_io_list(self):
        func_io_list = list()
        for func in self.net_proto.funcs:
            func_io = FuncProtoIO(func)
            func_io_list.append(func_io)
        return func_io_list

    def serialization(self, file_path):
        """
        Serialize to disk.
        """
        # self._get_graph_proto();
        with open(file_path, "wb") as f:
            f.write(self.net_proto.SerializeToString())
        f.close()

    def parse_from_string(self, file_path):
        """
        parser from optimized graph model
        """
        with open(file_path, "rb") as f:
            contents = f.read()
            self.net_proto.ParseFromString(contents)

    def __call__(self):
        return self.net_proto


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
