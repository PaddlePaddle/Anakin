import numpy as np
import os
from ..graph_io import *
from ..logger import *
from ..proto import *
import tensorflow as tf
from parse_tf_2_med import ParseTF2Med
from med_trans_util import MedTransAK


class TFParser:

	def __init__(self, fluid_config_dict):
		# anakin graph model io
		# config info
		self.ProtoPaths = fluid_config_dict['ProtoPaths']

		self.OutPuts = fluid_config_dict['OutPuts']
		if self.OutPuts is not None:
			self.OutPuts=[i for i in fluid_config_dict['OutPuts'].split(',')]

		self.med_trans_tool = MedTransAK()
		self.input_count=0

	def __call__(self):
		med_graph = self._conver_tf_2_med()
		if self.OutPuts is None:
			self.OutPuts=self._search_output_list(med_graph)
		anakin_graph = self._conver_med_2_anakin(med_graph)
		return anakin_graph

	def _conver_tf_2_med(self):
		parser = ParseTF2Med(self.ProtoPaths)
		return parser.parse()

	def _add_protonode(self, ak_graph,med_node):
		ak_type = med_node['ak_type']
		if ak_type is None:
			return
		nodeIO = NodeProtoIO()
		if med_node['ak_type']=='Input':
			nodeIO.set_name('input_'+str(self.input_count))
			self.input_count+=1
		else:
			nodeIO.set_name(med_node['name'])
		self.med_trans_tool.map_med_2_ak(nodeIO,med_node)
		ak_graph.add_node(nodeIO())
		if nodeIO().Op.name=='Input':
			ak_graph.add_in(nodeIO().name)

	def _search_output_list(self,graph):
		output_list=set()
		graph_cp=graph.copy()

		def recursive_search(node):

			if node.get('out_search_flat') is not None:
				return set()
			node['out_search_flat']=True
			outputs=node['output']
			result = set()
			if len(outputs)==0:
				result.add(node['name'])
			else:
				for i in outputs:
					result|=recursive_search(graph[i])

			return result


		for i in graph_cp.values():
			output_list|=recursive_search(i)
		return list(output_list)

	def _conver_med_2_anakin(self, med_graph):
		anakin_graph = GraphProtoIO()
		for node in med_graph.values():
			self._add_protonode(anakin_graph,node)
		anakin_graph.format_edge_from_nodes()
		for out_node_name in self.OutPuts:
			anakin_graph.add_out('output_'+out_node_name,out_node_name)
		return anakin_graph
