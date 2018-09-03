#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from utils import *
from proto import *
from logger import *
from frontend import RunServerOnGraph


class Graph(object):
    """
    Local abstract Graph architecture, it holds the real class GraphProto.
    It's used for parsering the useful info of GraphProto.
    """
    def __init__(self, config):
        """
        category: CAFFE, LEGO, PADDLE, TF, MXNET
        """
        self.save_file_path = config.SavePath + config.ResultName + ".anakin.bin"
        if config.framework == 'CAFFE':
            from kill_caffe import CaffeParser
            self.parser = CaffeParser(config.framework_config_dict)
        elif config.framework == 'PADDLE':
            pass
        elif config.framework == 'LEGO':
            pass
        elif config.framework == 'TF':
            pass
        elif config.framework == 'MXNET':
            pass
        elif config.framework == 'FLUID':
            from kill_fluid import FluidParser
            self.parser = FluidParser(config.framework_config_dict)
        else:
            raise NameError('ERROR: GrapProtoIO not support %s model.' % (config.framework))
        self.graph_io = self.parser()
        self.config = config

    def get_in(self):
        """
        Get input list of GraphProto (node name string list).
        """
        pass

    def get_out(self):
        """
        Get output list of GraphProto (node name string list).
        """
        pass

    def get_node_by_name(self):
        """
        Get the node according to it's name string.
        """
        pass

    def get_nodes(self):
        """
        Get node list.
        """
        pass

    def get_node_num(self):
        """
        Get node num in GraphProto.
        """
        pass

    def get_edge_num(self):
        """
        Get edge num.
        """
        pass

    def is_dege(self, node_name_0, node_name_1):
        """
        Judge if edge <node_name_0, node_name_1> is in GraphProto.
        """
        pass

    def get_version(self):
        """
        Get GraphProto version.
        """
        pass

    @RunServerOnGraph
    def run_with_server(self, ip="0.0.0.0", port=8888):
        """
        run the parser in web
        default web site: localhost:8888
        """
        return self.graph_io, self.config

    def serialization(self): 
        """
        serialize to disk
        """
        logger(verbose.WARNING).feed(" The model will be save to path: ", self.save_file_path)
        if not os.path.exists(os.path.dirname(self.save_file_path)):
            os.makedirs(os.path.dirname(self.save_file_path))
        self.graph_io.serialization(self.save_file_path)
