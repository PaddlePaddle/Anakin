import numpy as np
import os
from ..graph_io import *
from ..logger import *
from ..proto import *
import tensorflow as tf
from parse_tf_2_med import ParseTF2Med
from parse_med_2_ak import MedTransAK
from med_graph import MedGraphUtil, MedNodeUtil


class TFParser:

    def __init__(self, fluid_config_dict):
        # anakin graph model io
        # config info
        self.ProtoPaths = fluid_config_dict['ModelPath']

        self.OutPuts = fluid_config_dict['OutPuts']
        if self.OutPuts is not None:
            self.OutPuts = [i for i in fluid_config_dict['OutPuts'].split(',')]

        self.med_trans_tool = MedTransAK()
        self.input_count = 0

    def __call__(self):
        med_graph = self._conver_tf_2_med()
        if self.OutPuts is None:
            self.OutPuts = MedGraphUtil.search_output_list(med_graph)

        MedGraphUtil.solve(med_graph)
        anakin_graph = self._conver_med_2_anakin(med_graph)
        return anakin_graph

    def _conver_tf_2_med(self):
        '''
        convert tf graph to med graph
        :return:
        '''
        parser = ParseTF2Med(self.ProtoPaths)
        return parser.parse()

    def _add_protonode(self, ak_graph, med_node):
        '''
        add protobuf node in ak graph
        :param ak_graph:
        :param med_node:
        :return:
        '''
        ak_type = med_node['ak_type']
        if ak_type is None:
            return
        nodeIO = NodeProtoIO()
        nodeIO.set_name(med_node['name'])
        self.med_trans_tool.map_med_2_ak(nodeIO, med_node)
        ak_graph.add_node(nodeIO())
        if nodeIO().Op.name == 'Input':
            ak_graph.add_in(nodeIO().name)

    def _conver_med_2_anakin(self, med_graph):
        '''
        convert med graph to ak graph
        :param med_graph:
        :return:
        '''
        anakin_graph = GraphProtoIO()
        for node in med_graph.values():
            self._add_protonode(anakin_graph, node)
        anakin_graph.format_edge_from_nodes()
        for out_node_name in self.OutPuts:
            anakin_graph.add_out('output_' + out_node_name, out_node_name)
        return anakin_graph
