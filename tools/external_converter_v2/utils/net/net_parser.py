#! /usr/bin/env python
# -*- coding: utf-8 -*-

from net_io import NetProtoIO

class NetParser:
    """
    """
    def __init__(self, config):
        '''
        '''
        assert 'NET' in config.DebugConfig.keys()
        self.config = config.DebugConfig['NET']
        self.load_list = self.config['LoadPaths']
        self.save_format = self.config['SaveFormat']
        self.net_io = NetProtoIO()

    def __str__(self):
        return self.net_io.net_proto.__str__()

    def storage(self):
        storager = NetStorage(self.net_io)
        storager()

    def load(self):
        for load_path in self.load_list:
            self.net_io.merge_from_string(load_path)

    def save(self):
        if self.save_format == 'binary':
            self.net_io.serialization(self.config['SavePath'])
        elif self.save_format == 'text':
            self.net_io.save_txt(self.config['SavePath'])

    def __call__(self):
        self.load()
        self.storage()
        self.save()


class NetStorage(object):
    """
    """
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
