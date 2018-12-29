#! /usr/bin/env python
# -*- coding: utf-8 -*-

from net_io import NetProtoIO


class NetParser:

    def __init__(self, net_config_dict):
        '''
        '''
        assert net_config_dict is not None
        self.load_list = net_config_dict['LoadPaths']
        self.save_directory = net_config_dict['SavePath']
        self.net_proto_list = list()
        self.load_nets()

    def __call__(self):
        pass

    def load_nets(self):
        for load_path in self.load_list:
            Net
            net_proto.parse_from_string(load_path)
            self.
        pass

    def save_nets(self):
        pass





