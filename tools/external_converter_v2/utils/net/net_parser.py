#! /usr/bin/env python
# -*- coding: utf-8 -*-

from net_io import NetProtoIO
from storage import NetStorage

class Net:
    """
    """
    def __init__(self, config):
        '''
        '''
        self.config = config.DebugConfig
        self.parser = NetParser(self.config)
        self.net_io = self.parser()

    def __str__(self):
        return net_io.net_proto.__str__()

    def storage(self):
        storager = NetStorage(net_io)
        storager()

    def serialization(self):
        self.net_io.serialization(self.config['SavePath'])


class NetParser:
    """
    """
    def __init__(self, net_config_dict):
        '''
        '''
        assert net_config_dict is not None
        self.load_list = net_config_dict['LoadPaths']
        self.net_io = NetProtoIO()

    def load_nets(self):
        for load_path in self.load_list:
            self.net_io.merge_from_string(load_path)

    def __call__(self):
        self.load_nets()
        return self.net_io

