import tensorflow as tf
import numpy as np
from tensorflow.core.framework import types_pb2, tensor_pb2
import logging as log
import collections
from tf_trans_util import *


class ParseTF2Med:
    def __init__(self, tf_forzen_pb_path):
        self.model_path = tf_forzen_pb_path

    def _debug_nodes(self, nodes):
        '''
        print debug info and exit
        :param nodes:
        :return:
        '''
        for i in nodes.values():
            print(i['name'], i['input'], i['output'], i['out_shape'])
        print('debug end')
        exit()

    def _parse_tf_node(self, tf_graph, shape_override):
        """
        Load tensorflow graph into an raw med graph
        """

        # ignore the following attributes
        ignored_attr = ["unknown_rank", "_class", "Tidx", "Tshape", "use_cudnn_on_gpu", "Index",
                        "Tpaddings", "TI", "Tparams", "Tindices", "Tlen", "Tdim", "dynamic_size", "element_shape",
                        "Tmultiples", "output_dtype", "Tblock_shape", "Tcrops", "index_type", "Taxis", "U",
                        "maxval", "Tout"]
        # some stats
        op_cnt = collections.Counter()
        attr_cnt = collections.Counter()
        anakin_nodes = {}
        dtypes = {}

        # find outputs
        ops = tf_graph.get_operations()

        tensor_shape = {}
        for node in ops:
            for out in node.outputs:
                tensor_shape[out.name] = out.get_shape().as_list()

        # minimal conversion of attributes
        for node in ops:
            attr = {}
            takeit = True
            op_cnt[node.type] += 1

            for a in node.node_def.attr:
                a = str(a)
                attr_cnt[a] += 1
                if a == "dtype":
                    attr[a] = map_tf_dtype(node.get_attr("dtype"))
                elif a == "T":
                    dtype = node.get_attr("T")
                    if dtype:
                        if not isinstance(dtype, list):
                            dtypes[node.name] = map_tf_dtype(dtype)
                elif a in ["output_type", "output_dtype", "out_type"]:
                    attr[a] = map_tf_dtype(node.get_attr(a))
                elif a == "shape":
                    attr[a] = get_shape(node)
                elif a == "Tperm":
                    pass
                elif a == "_output_shapes":
                    attr[a] = get_shape(node)
                elif a == "value":
                    anakin_tensor = tf_to_anakin_tensor(node.get_attr(a))
                    attr[a] = anakin_tensor
                elif a == "DstT":
                    attr["to"] = map_tf_dtype(node.get_attr("DstT"))
                elif a == "SrcT":
                    continue
                elif a in ignored_attr:
                    continue
                else:
                    attr[a] = node.get_attr(a)

            if takeit:
                try:
                    input_names = [i.name for i in node.inputs]
                    output_names = [i.name for i in node.outputs]
                    anakin_nodes[node.name] = {'name': node.name, 'type': node.type,
                                               'input': input_names,
                                               'output': output_names,
                                               'tf_attr': attr, 'visted': False,
                                               'ak_type': None, 'ak_attr': {}}
                except Exception as ex:
                    log.error("pass1 convert failed for %s, ex=%s", node, ex)
                    raise

        self._fix_self_output(anakin_nodes, tensor_shape)

        return anakin_nodes

    def _fix_self_output(self, nodes, tensor_shape_dict):
        '''
        convert tensor connection to op connection
        :param nodes:
        :param tensor_shape_dict:
        :return:
        '''
        out2nodename = {}
        for node in nodes.values():
            for out_name in node['output']:
                if out2nodename.get(out_name) is None:
                    out2nodename[out_name] = [node['name']]
                else:
                    out2nodename[out_name].append(node['name'])

        in2nodename = {}
        for node in nodes.values():
            for in_name in node['input']:
                if in2nodename.get(in_name) is None:
                    in2nodename[in_name] = [node['name']]
                else:
                    in2nodename[in_name].append(node['name'])

        for node in nodes.values():
            new_output = []
            new_input = []

            for tensor_name in node['output']:
                if in2nodename.get(tensor_name) is not None:
                    new_output += [{'name': op_name, 'shape': tensor_shape_dict[tensor_name]} for op_name in
                                   in2nodename[tensor_name]]

            for tensor_name in node['input']:
                if out2nodename.get(tensor_name) is not None:
                    new_input += [{'name': op_name, 'shape': tensor_shape_dict[tensor_name]} for op_name in
                                  out2nodename[tensor_name]]

            node['output'] = new_output
            node['input'] = new_input

    def _parse_tf_graph(self, nodes):
        '''
        conver op in tf graph to med graph
        :param nodes:
        :return:
        '''

        def all_search(graph, table):
            for tf_node in graph.values():
                if tf_node['visted']:
                    continue
                type_name = tf_node['type']
                if table.get(type_name) != None:
                    table[type_name](tf_node, graph)

        def all_search_fix(graph, table):
            for tf_node in graph.values():
                type_name = tf_node['ak_type']
                if table.get(type_name) != None:
                    table[type_name](tf_node, graph)

        all_search(nodes, {'Identity': parse_Identity,
                           'Placeholder': parse_Placeholder,
                           'Shape': parse_Shape,
                           'StridedSlice': parse_slim_flatten
                           })

        all_search(nodes, {'Reshape': parse_fusionReshape, })

        all_search(nodes, {'MatMul': parse_MatMul,
                           'Conv2D': parse_Conv2D,
                           'DepthwiseConv2dNative': parse_Conv2D,
                           'FusedBatchNorm': parse_BatchNorm,
                           'Rsqrt': parse_CustmerBatchNorm, })

        all_search(nodes, {'Add': parse_Add,
                           'AvgPool': parse_Pooling,
                           'ConcatV2': parse_Concat,
                           'MaxPool': parse_Pooling,
                           'Mean': parse_Mean,
                           'Pad': parse_Pad,
                           'Relu': parse_Act,
                           'Relu6': parse_Act,
                           'Reshape': parse_Reshape,
                           'Squeeze': parse_Squeeze,
                           'Softmax': parse_Softmax,
                           'Transpose': parse_Transpose
                           })

        all_search_fix(nodes, {'Dense': fix_Dense})

        return nodes

    def parse(self):
        '''
        entrance to load tf graph and convert it to med graph
        :return:
        '''
        tf_graph = load_graph(self.model_path)
        nodes = self._parse_tf_node(tf_graph, {})

        med_graph = self._parse_tf_graph(nodes)
        filter_graph = {i: med_graph[i] for i in med_graph.keys() if med_graph[i]['ak_type'] is not None}
        for node in filter_graph.values():
            node['input'] = [i for i in node['input'] if filter_graph.get(i['name']) is not None]
            node['output'] = [i for i in node['output'] if filter_graph.get(i['name']) is not None]
        return filter_graph
