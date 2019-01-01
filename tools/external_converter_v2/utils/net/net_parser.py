#! /usr/bin/env python
# -*- coding: utf-8 -*-

from net_io import NetProtoIO
from storage import NetStorage

class Net:

    def __init__(self, config):
        '''
        '''
        self.config = config.DebugConfig
        self.parser = NetParser(self.config)
        self.net_io = self.parser()

    def serialization(self):
        self.net_io.serialization(self.config['SavePath'])

class NetParser:

    def __init__(self, net_config_dict):
        '''
        '''
        assert net_config_dict is not None
        self.load_list = net_config_dict['LoadPaths']
        self.net_io_list = list()

    def load_nets(self):
        for load_path in self.load_list:
            net_proto = NetProtoIO()
            net_proto.parse_from_string(load_path)
            self.net_io_list.append(net_proto)

    def storage_net(self):
        for net_io in self.net_io_list:
            storager = NetStorage(net_io)
            storager()

    def print_net(self):
        for net_io in self.net_io_list:
            print net_io()

    def parser(self):
        self.load_nets()
        self.storage_net()

    def __call__(self):
        self.parser()
        #self.print_net()
