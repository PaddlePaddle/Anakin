#! /usr/bin/env python
# -*- coding: utf-8 -*-

from net_io import NetProtoIO

class Net:
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
        self.load()

    def __str__(self):
        return self.net_io.net_proto.__str__()

    def parse(self):
        parser = NetParser(self.net_io, self.config)
        parser.net_reset_nodes()
        parser.nets_slice()
        parser.save_funcs()

    def load(self):
        for load_path in self.load_list:
            self.net_io.merge_from_string(load_path)

    def __call__(self):
        return self.net_io


class NetParser(object):
    """
    """
    def __init__(self, net_io, config):
        # reset node in funcs
        self.config = config
        self.net_io_in = net_io_in
        self.graph_io = self.net_io_in.graph_io()
        self.func_io_list = self.net_io_in.func_io_list()
        # funcs slice
        self.nets_io_out = list()
        self.funcs = dict()
        self.save_path = self.config['SavePath']

    def _clear_graph(self):
        self.net_io_in.clear_graph()

    def _funcs_dict(self):
        for func_io in self.func_io_list:
            func_type = func_io.get_type()
            if func_type not in self.funcs.keys():
                self.funcs[func_type] = list()
            self.funcs[func_type].append(func_io)

    def net_reset_nodes(self):
        for func_io in self.func_io_list:
            func_name = func_io.get_name()
            node_io = self.graph_io.get_node_io(func_name)
            func_io.reset_node_io(node_io)
        self._clear_graph()
        self.nets_io_out.append(self.net_io_in)
        return self.nets_io_out

    def nets_slice(self):
        self.nets_io_out = list()
        self._funcs_dict()
        for func_type in self.funcs.keys():
            net = NetProtoIO()
            net.set_name(func_type)
            funcs_list = self.funcs[func_type]
            for func in funcs_list:
                net.add_func(func)
            self.nets_io_out.append(net)
        return self.nets_io_out

    def save_funcs(self):
        for net_io_out in self.nets_io_out:
            net_io_out.save(self.save_path)

