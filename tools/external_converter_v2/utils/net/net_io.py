#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from parser.proto import net_pb2
from parser.graph_io import GraphProtoIO
from google.protobuf import text_format


class FuncProtoIO(object):
    """
    Func io class of FuncProto.
    """

    def __init__(self, proto=None):
        """
        Initial the FuncProtoIO object.
        """
        self.func_proto = None
        if proto is None:
            self.func_proto = net_pb2.FuncProto()
        else:
            self.func_proto = proto

    def get_name(self):
        """
        Get the name of func_proto.
        """
        return self.func_proto.name

    def set_name(self, name):
        """
        Set the name of func_proto.
        """
        self.func_proto.name = name

    def get_type(self):
        """
        Get the type of func_proto.
        """
        return self.func_proto.type

    def set_type(self, type_value):
        """
        Set the type of func_proto.
        """
        self.func_proto.type = type_value

    def get_node_io(self):
        """
        Get the node io of this object.
        """
        node_io = NodeProtoIO(self.func_proto.node_info)
        return node_io

    def reset_node_io(self, node_io):
        """
        Reset the node io of this object.
        """
        node_proto = node_io()
        self.func_proto.node_info.CopyFrom(node_proto)

    def __call__(self):
        """
        Return func_proto.
        """
        return self.func_proto


class NetProtoIO(object):
    """
    Net io class of NetProto.
    """

    def __init__(self, proto=None):
        """
        Init the NetProtoIO object.
        """
        self.net_proto = None
        if proto is None:
            self.net_proto = net_pb2.NetProto()
        else:
            self.net_proto = proto

    def graph_io(self):
        """
        Generate the graph io.
        """
        graph_io = GraphProtoIO(self.net_proto.graph)
        return graph_io

    def clear_graph(self):
        """
        Clear the graph of net proto.
        """
        self.net_proto.graph.Clear()

    def get_name(self):
        """
        Get the name of net_proto.
        """
        return self.net_proto.name

    def set_name(self, net_name):
        """
        Set the name of net_proto.
        """
        self.net_proto.name = net_name

    def add_func(self, func=None):
        """
        Add a func proto.
        """
        if func is None:
            func = net_pb2.FuncProto()
        self.net_proto.funcs.extend([func])

    def func_io_list(self):
        """
        Add func io list.
        """
        func_io_list = list()
        for func in self.net_proto.funcs:
            func_io = FuncProtoIO(func)
            func_io_list.append(func_io)
        return func_io_list

    def save(self, file_path, use_txt=True, use_net_name=True):
        """
        Save the Net proto.
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

    def merge_from_io(self, net_io):
        """
        Merge proto from io.
        """
        self.net_proto.MergeFrom(net_io.net_proto)

    def merge_from_string(self, file_path):
        """
        parser from optimized graph model
        """
        with open(file_path, "rb") as f:
            contents = f.read()
            self.net_proto.MergeFromString(contents)

    def __call__(self):
        """
        Return the net_proto.
        """
        return self.net_proto
