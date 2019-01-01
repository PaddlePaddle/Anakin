#! /usr/bin/env python
# -*- coding: utf-8 -*-

class NetStorage(object):

    def __init__(self, net_io):
        self.net_io = net_io
        self.graph_io = self.net_io.graph_io()
        self.func_io_list = self.net_io.func_io_list()

    def reset_node_in_func(self):
        for func_io in self.func_io_list:
            func_name = func_io.get_name()
            node_io = self.graph_io.get_node_io(func_name)
            func_io.reset_node_io(node_io)

    def clear_graph(self):
        self.net_io.clear_graph()

    def net_storage(self):
        self.reset_node_in_func()
        self.clear_graph()

    def __call__(self):
        self.net_storage()
