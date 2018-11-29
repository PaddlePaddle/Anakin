import onnx
import numpy as np
#from tensorflow.core.framework import types_pb2, tensor_pb2
import logging as log
import collections
from onnx_trans_utils import *

class ParseOnnxToMed:
    def __init__(self, onnx_model_path):
        self.model_path = onnx_model_path

    def _parse_onnx_node(self, onnx_graph, shape_override):
        """
        Load onnx graph and parse node
        :param onnx_graph:
        :param shape_override:
        :return:
        """

        # ignore the following attributes
        ignored_attr = ["unknown_rank", "_class", "Tidx", "Tshape", "use_cudnn_on_gpu", "Index",
                        "Tpaddings", "TI", "Tparams", "Tindices", "Tlen", "Tdim",
                        "dynamic_size", "element_shape", "Tmultiples", "output_dtype",
                        "Tblock_shape", "Tcrops", "index_type", "Taxis", "U",
                        "maxval", "Tout"]
        # some stats
        op_cnt = collections.Counter()
        attr_cnt = collections.Counter()
        anakin_nodes = {}
        dtypes = {}

        # find ops
        ops = onnx_graph.node

        # minimal conversion of attributes
        for node in ops:
            attr = {}
            takeit = True

            for a in node.attribute:
                attr_cnt[a.name] += 1
                if a.type == 1: ##FLAOT
                    attr[a.name] = a.f
                elif a.type == 2: #INT
                    attr[a.name] = int(a.i)
                elif a.type == 6: #FLOATS
                    val_list = []
                    for val in a.floats:
                        val_list.append(val)
                    attr[a.name] = val_list
                elif a.type == 7: #INTS
                    val_list = []
                    #print 'type: ', a.name, type(a.ints[0])
                    for val in a.ints:
                        val_list.append(int(val))
                    attr[a.name] = val_list
                else:
                    attr[a.name] = a.auto_pad

            if takeit:
                try:
                    #input_names = [i for i in node.input]
                    #output_names = [i for i in node.output]
                    # if node.name == '':
                    #    node.name = node.output[0]
                    name = node.name #name + '_' +
                    node.name = name + '_' + str(node.op_type) + '_' + str(op_cnt[node.op_type])
                    op_cnt[node.op_type] += 1
                    #print node_name
                    #node_name = node.output[0];
                    anakin_nodes[node.name] = {'name': node.name, 'type': node.op_type,
                                               'input': [str(i) for i in node.input],
                                               'output': [str(i) for i in node.output],
                                               'onnx_attr': attr, 'visited': False, 'name:': False,
                                               'shape': None, 'ak_type': None, 'ak_attr': {}}
                except Exception as ex:
                    log.error("pass1 convert failed for %s, ex=%s", node, ex)
                    raise
        #print 'anakin_node', anakin_nodes
       # exit()
        #weights and bias
        graph = onnx_graph.initializer
        # print 'weights: ', graph
        weights = {}
        for init_ptr in graph:
            print 'init_ptr: ', init_ptr.name
           #  print ('onnx_to_anakin_tensor: ')
            [data, shape, dtype] = onnx_to_anakin_tensor(init_ptr)
            # print ('end')
            anakin_tensor = {}
            anakin_tensor['shape'] = shape
            anakin_tensor['data'] = data
            anakin_tensor['dtype'] = dtype
            #attr[init_ptr.name] = anakin_tensor
            #anakin_nodes[init_ptr.name] = {'name': init_ptr.name, 'onnx_attr': attr, 'visited': False,
             #                               'shape':None, 'ak_type': None, 'ak_attr': {}}
            weights[init_ptr.name] = anakin_tensor
            # if init_ptr.name == 'OC2_DUMMY_3':
            #     print (init_ptr, type(data), data, shape, dtype)
            #     exit(0)
           # print 'name: ', init_ptr.name, dtype, shape,

            #print 'tensor: ',  anakin_tensor
            #exit()

        input_name = onnx_graph.input
        inputs = {}
        input_shape = {}
        in_cnt = 0
        # print '--------input---------'
        # print input_name
        for input_a in input_name:
            shape = []
            for dim in input_a.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            #attr["shape"] = shape
            if input_a.name.startswith('data'):
                inputs[input_a.name] = shape
                output_node = []
                #print 'input: ', input_a.name
                for node in anakin_nodes.values():
                    for name in node['input']:
                        if name == input_a.name:
                            output_node.append(name) #(node_name)
                #print 'out: ', output_node
                node_name = str('input') + '_' + str(in_cnt)
                # change inputs name in anakin nodes
                '''
                for node in anakin_nodes.values():
                    in_name = node['input']
                    for i in range(len(in_name)):
                        if in_name[i] == input_a.name:
                            in_name[i] = node_name
                '''

                anakin_nodes[node_name] = {'name': node_name, 'type': 'Input',
                                            'input': [], 'output': output_node,
                                            'onnx_attr': {}, 'visited': True,
                                            'shape': shape, 'ak_type': 'Input',
                                            'ak_attr': {'shape': shape}}

                in_cnt += 1
            else:
                #print 'name: ', input_a.name
                input_shape[input_a.name] = shape

        output_name = onnx_graph.output
        outputs = {}
        for output_a in output_name:
            shape = []
            for dim in output_a.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            outputs[output_a.name] = shape
            input_node = []
            for node in anakin_nodes.values():
                for name in node['output']:
                    if name == output_a.name:
                        input_node.append(name)

            anakin_nodes[output_a.name] = {'name': output_a.name, 'type': 'Output',
                                          'input': input_node,
                                          'output': [], 'onnx_attr': {}, 'visited': True,
                                          'shape': shape, 'ak_type': None, 'ak_attr': {}}
        #print 'weights', len(weights)
        #print 'weights', weights
        '''
        for node_name in anakin_nodes.keys():
            for node_out in output_name:
                if node_name == node_out:
                    anakin_nodes[node_name]['output'] = []
        '''
        # change inputs outputs name
        self._change_inout_nodename(anakin_nodes, weights)
        # print 'anakin_node', anakin_nodes

        output_node = {}
        for node in anakin_nodes.values():
            for out_name in node['output']:
                if out_name in outputs:
                    output_node[node['name']] = outputs[out_name]
                    # delete output
                    node['output'].pop()
            outnode = node['output']
            for i in range(len(outnode)):
                if outnode[i] in outputs:
                    outnode.pop(i)


        #print 'inputs', inputs
        #print 'outputs', outputs
        return [anakin_nodes, weights, outputs, output_node]

    def _change_inout_nodename(self, nodes, weights):
        """
        convert tensor connection to op connection
        :param nodes:
        :param weights:
        :return:
        """
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

        # print 'in2node_name', in2nodename
        # print 'out2node_name', out2nodename
        # print 'shape', shape

        for node in nodes.values():
            # print 'in:', node['input']
            # print 'out:', node['output']
            new_output = []
            new_input = []

            for out_name in node['output']:
                if in2nodename.get(out_name) is not None:
                    new_output += [op_name for op_name in in2nodename[out_name]]
            for in_name in node['input']:
                if out2nodename.get(in_name) is not None:
                    new_input += [op_name for op_name in out2nodename[in_name]]
                # bias and weights
                if weights.has_key(in_name):
                    new_input += [in_name]


            node['input'] = new_input
            node['output'] = new_output
            # print 'node:', node['name']
            # print 'in:', node['input']
            # print 'out:', node['output']

    def _parse_onnx_graph(self, nodes, weights):
        """
        parse onnx
        :param nodes:
        :param weights:
        :return:
        """
        # out2nodename = {i['name']:[] for i in nodes}
        #self._fix_self_output(nodes)

        def all_search(graph, table):
            """
            search the graph
            :param graph:
            :param table:
            :return:
            """
            for onnx_node in graph.values():
                if onnx_node['visited']:
                    continue
                type_name = onnx_node['type']
                if table.get(type_name) != None:
                    table[type_name](onnx_node, weights, graph)

        all_search(nodes, {'Conv': parse_Conv,
                           'Gemm': parse_Gemm,
                           'BatchNormalization': parse_BatchNorm})

        all_search(nodes, {'Concat': parse_Concat})

        all_search(nodes, {'Add': parse_Add,
                           'LRN': parse_Lrn,
                           'Softmax': parse_Softmax,
                           'Dropout': parse_Dropout,
                           'Relu': parse_Act,
                           'MaxPool': parse_Pooling,
                           'GlobalAveragePool': parse_Pooling,
                           'Reshape': parse_Reshape})
        #nodes = rm_weight_node(nodes, weights)
        #print 'anakin_node: ', nodes
        return nodes

    def parse(self):
        """
        parse onnx
        :return:
        """
        onnx_model = onnx.load(self.model_path)
        onnx_graph = onnx_model.graph
        [nodes, weights, outputs, output_node] = self._parse_onnx_node(onnx_graph, {})
        print ('onnx_node')
        for node in nodes.values():
            print(node['name'], node['input'], node['output'])
        # exit()
        print ('-------------------------------')
        med_graph = self._parse_onnx_graph(nodes, weights)
        print ('weights or bias shape:')
        for wei in weights:
            wei_node = weights[wei]
            print(wei, wei_node['shape'])
        print ('-------------------------------')
       # filter_graph={i:med_graph[i] for i in med_graph.keys() if med_graph[i]['ak_type'] is not None}
        return med_graph, output_node #filter_graph, outputs
