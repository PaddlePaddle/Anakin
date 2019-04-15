import numpy as np
import os
from ..graph_io import *
from ..logger import *
from ..proto import *
import onnx
from onnx_graph import ParseOnnxToMed
from med_trans_util import MedTransAK
from med_graph import MedGraphUtil, MedNodeUtil

class OnnxParser:
    """
    onnx parse begin
    """
    def __init__(self, onnx_config_dict):
		# anakin graph model io
		# config info
		# print 'onnx_config_dict', onnx_config_dict

        # self.ProtoPaths = onnx_config_dict['ProtoPaths']
        self.OnnxPaths = onnx_config_dict['ModelPath']
        if onnx_config_dict['TxtPath'] == '':
            self.txtPaths = None
        else:
            self.txtPaths = onnx_config_dict['TxtPath']
        self.med_trans_tool = MedTransAK()
        self.input_count = 0

    def __call__(self):
        [med_graph, outputs] = self._conver_onnx_2_med()
        self.Output = outputs
        MedGraphUtil.solve(med_graph)
        anakin_graph = self._conver_med_2_anakin(med_graph)
        return anakin_graph


    def _conver_onnx_2_med(self):
        """
        convert onnx to med graph
        :return:
        """
        parser = ParseOnnxToMed(self.OnnxPaths, self.txtPaths)
        return parser.parse()

    def _add_protonode(self, ak_graph, med_node):
        """
        add med node to anakin graph
        :param ak_graph:
        :param med_node:
        :return:
        """
        ak_type = med_node['ak_type']
		# print '_add_protonode', med_node['name'], ak_type
        if ak_type is None:
			# print 'ak_type'
            return
        nodeIO = NodeProtoIO()
        if med_node['ak_type'] == 'Input':
            nodeIO.set_name('input_' + str(self.input_count))
            self.input_count += 1
        else:
            nodeIO.set_name(med_node['name'])
        self.med_trans_tool.map_med_2_ak(nodeIO, med_node)
        ak_graph.add_node(nodeIO())
        if nodeIO().Op.name == 'Input':
           ak_graph.add_in(nodeIO().name)
		#print 'node: ', med_node['name']

    def _search_output_list(self, graph):
        """
        search output list
        :param graph:
        :return:
        """
        output_list=set()
        graph_cp=graph.copy()

        def recursive_search(node):
            """
            recursive search
            :param node:
            :return:
            """
            if node.get('out_search_flat') is not None:
                return set()
            node['out_search_flat']=True
            outputs=node['output']
            result = set()
            if len(outputs) == 0:
                result.add(node['name'])
            else:
                for i in outputs:
            	    result |= recursive_search(graph[i])
            return result


        for i in graph_cp.values():
            output_list |= recursive_search(i)
        return list(output_list)

    def _conver_med_2_anakin(self, med_graph):
        """
        convert med graph too anakin graph
        :param med_graph:
        :return:
        """
        anakin_graph = GraphProtoIO()
		#print 'med_graph: ', med_graph
        for node in med_graph.values():
            self._add_protonode(anakin_graph, node)

        print '*************anakin**************'
        anakin_graph.format_edge_from_nodes()
        for out_node_name in self.Output:
            anakin_graph.add_out('output_' + out_node_name, out_node_name)
            print 'out', out_node_name
        return anakin_graph
