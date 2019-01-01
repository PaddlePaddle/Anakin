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
        assert 'NET' in config.DebugConfig.keys()
        self.config = config.DebugConfig['NET']
        self.parser = NetParser(self.config)
        self.net_io = self.parser()
        self.save_format = self.config['SaveFormat']

    def __str__(self):
        return self.net_io.net_proto.__str__()

    def storage(self):
        storager = NetStorage(self.net_io)
        storager()

    def save(self):
        if self.save_format == 'binary':
            self.net_io.serialization(self.config['SavePath'])
        elif self.save_format == 'text':
            self.net_io.save_txt(self.config['SavePath'])

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

