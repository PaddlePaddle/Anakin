#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from utils import *
from proto import *
from logger import *
from frontend import RunServerOnGraph
from prettytable import PrettyTable


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
            from caffe import CaffeParser
            self.parser = CaffeParser(config.framework_config_dict)
        elif config.framework == 'PADDLE':
            pass
        elif config.framework == 'LEGO':
            from lego import LegoParser_test
            self.parser = LegoParser_test(config.framework_config_dict)
        elif config.framework == 'TENSORFLOW':
            from tensorflow import TFParser
            self.parser=TFParser(config.framework_config_dict)
        elif config.framework == 'MXNET':
            pass
        elif config.framework == 'FLUID':
            from fluid import FluidParser
            self.parser = FluidParser(config.framework_config_dict)
        else:
            raise NameError('ERROR: GrapProtoIO not support %s model.' % (config.framework))
        self.graph_io = self.parser()
        self.config = config

    def ins(self):
        """
        Get input list of GraphProto (node name string list).
        """
        return self.graph_io.ins()

    def outs(self):
        """
        Get output list of GraphProto (node name string list).
        """
        return self.graph_io.outs()

    def get_node_by_name(self, node_name):
        """
        Get the node according to it's name string.
        """
        return self.graph_io.find_node_proto(node_name)

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

    def info_table(self):
        """
        print input table.
        """
        tables = list()
        in_table = PrettyTable(["Input Name", "Shape", "Alias", "Data Type"])
        out_table = PrettyTable(["Output Name"])

        def ins_attr():
            """
            get inputs attr.
            """
            ins = list()
            for graph_in in self.ins():
                attr = dict()
                proto = self.get_node_by_name(graph_in)
                attr['name'] = graph_in
                attr['shape'] = proto.attr['input_shape'].cache_list.i
                attr['type'] = proto.attr['data_type'].s
                attr['alias'] = proto.attr['alias'].s
                ins.append(attr)
            return ins

        for attr in ins_attr():
            in_table.add_row([attr['name'], attr['shape'], attr['alias'], attr['type']])
        for out_name in self.outs():
            out_table.add_row([out_name])

        print in_table
        print out_table
