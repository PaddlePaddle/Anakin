#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")

from parser.proto import net_pb2

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

    def graph(self):
        return self.net_proto.graph

    def clear_graph(self):
        self.net_proto.graph.Clear()

    def funcs(self):
        return self.net_proto.funcs

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

    def node(self):
        return self.func_proto.node_info

    def fill_node(self, node):
        self.func_proto.node_info = node

    def __call__(self):
        return self.func_proto


class NodeProtoIO(object):
    """
    Node io class of FuncProto.
    """

    def __init__(self, proto=None):
        """
        """
        self.node_proto = None
        if proto is None:
            self.node_proto = net_pb2.NodeProto()
        else:
            self.node_proto = proto

    def get_name(self):
        return self.node_proto.name

    def __call__(self):
        return self.func_proto
