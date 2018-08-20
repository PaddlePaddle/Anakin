import tensorflow as tf
import numpy as np
from tensorflow.core.framework import types_pb2, tensor_pb2
import logging as log
import collections
from tf_trans_util import *
class ParseTF2Med:
    def __init__(self,tf_forzen_pb_path):
        self.model_path=tf_forzen_pb_path



    def _parse_tf_node(self,tf_graph,shape_override):


        """
        Load tensorflow graph into an onnx graph with minimal rewrites so
        we can use the onnx graph as intermediate graph.
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
        output_shapes = {}
        dtypes = {}

        # find outputs
        ops = tf_graph.get_operations()


        # create dict with output to shape mappings
        for node in ops:
            for out in node.outputs:
                shape = shape_override.get(out.name)
                if shape is None:
                    try:
                        shape = out.get_shape().as_list()
                    except Exception as ex:
                        shape = []
                if shape == []:
                    print(out.name, '== []')
                dtypes[out.name] = map_tf_dtype(out.dtype)
                output_shapes[out.name[:out.name.find(':')]] = shape


        # minimal conversion of attributes
        for node in ops:
            attr = {}
            takeit = True
            op_cnt[node.type] += 1

            for a in node.node_def.attr:
                a=str(a)
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
                    # input_names = [i.name[:i.name.find(':')] for i in node.inputs]
                    # output_names = [i.name[:i.name.find(':')] for i in node.outputs]
                    input_names = [i.name for i in node.inputs]
                    output_names = [i.name for i in node.outputs]
                    anakin_nodes[node.name] = {'name': node.name, 'type': node.type, 'input': input_names,
                                               'output': output_names, 'tf_attr': attr, 'visted': False,
                                               'out_shape': output_shapes[node.name],
                                               'ak_type': None, 'ak_attr': {}}
                except Exception as ex:
                    log.error("pass1 convert failed for %s, ex=%s", node, ex)
                    raise
        return anakin_nodes

    def _fix_self_output(self,nodes):
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
            new_output=[]
            new_input=[]
            for i in node['output']:
                if in2nodename.get(i) is not None:
                    new_output=new_output+in2nodename[i]

            for i in node['input']:
                if out2nodename.get(i) is not None:
                    new_input=new_input+out2nodename[i]
            node['output']=new_output
            node['input']=new_input

    def _parse_tf_graph(self,nodes):
        # out2nodename = {i['name']:[] for i in nodes}
        self._fix_self_output(nodes)



        def all_search(graph, table):
            for tf_node in graph.values():
                if tf_node['visted']:
                    continue
                type_name = tf_node['type']
                if table.get(type_name) != None:
                    table[type_name](tf_node, graph)

        all_search(nodes, {'Identity': parse_Identity,
                           'Placeholder': parse_Placeholder})


        all_search(nodes, {'MatMul': parse_MatMul,
                           'Conv2D': parse_Conv2D})
        all_search(nodes, {'Reshape': parse_Reshape,
                           'Relu': parse_Act,
                           'MaxPool': parse_Pooling})
        return nodes


    def parse(self):
        tf_graph = load_graph(self.model_path)
        nodes = self._parse_tf_node(tf_graph, {})
        # for node in nodes.values():
        #     print(node['name'],node['input'],node['output'])
        # exit()
        med_graph=self._parse_tf_graph(nodes)
        filter_graph={i:med_graph[i] for i in med_graph.keys() if med_graph[i]['ak_type'] is not None}
        return filter_graph