import numpy as np

class MedNodeUtil:

    @staticmethod
    def new_med_node():
        """
        instance of empty standard med graph node
        :return:
        """
        return {'name': None, 'ak_type': None, 'input': [], 'output': [],
                'ak_attr': {}, 'type': None,
                'med_visted': False}

    @staticmethod
    def replace_name_with_list(origin_list, name, replace_list):
        """
        replace name in input or output with replace_list
        :param origin_list:
        :param name:
        :param replace_list:
        :return:
        """
        new_list = []
        for index, ori_object in enumerate(origin_list):
            if ori_object == name:
                new_list = new_list + replace_list + origin_list[index + 1:]
                break
            else:
                new_list.append(ori_object)
        if name in new_list:
            raise Exception('circle error')
        return new_list

    @staticmethod
    def retain_input(node, input_list):
        """
        remove node input except element in input_list
        :param node:
        :param input_list:
        :return:
        """
        new_input = []
        for index, node_object in enumerate(node['input']):
            if node_object in input_list:
                input_list.remove(node_object)
                new_input.append(node_object)
        node['input'] = new_input

    @staticmethod
    def redirecto_outputs_input_to_this(node, graph, this_name):
        """
        get node_x in node`s outputs
        make node_x`s inputs reference to node
        :param node:
        :param graph:
        :param this_name:
        :param this_shape:
        :return:
        """
        for i in node['output']:
            tar_node = graph[i]
            tar_node['input'] = MedNodeUtil.replace_name_with_list(tar_node['input'],
                                                                   node['name'],
                                                                   [this_name])


MedGraph_Input_Cnt = 0


class MedGraphUtil:
    """
    MEdGraph utils
    """
    @staticmethod
    def append_node(father_node, son_node, graph):
        """
        add the son_node after father_node in graph
        :param father_node:
        :param son_node:
        :param graph:
        :return:
        """
        # print ('father_node', father_node['name'], father_node['input'], father_node['output'])
        output = father_node['output']
        #son_shape = output[0]['shape']
        son_node['input'] = [father_node['name']]
        son_node['output'] = output
        father_node['output'] = [son_node['name']]
        for i in output:
            out_node = graph[i]
            out_node['input'] = MedNodeUtil.replace_name_with_list(out_node['input'],
                                                                   father_node['name'],
                                                                   [son_node['name']])
            # print ('out_node: ', out_node['name'], out_node['input'])
        graph[son_node['name']] = son_node
        # print ('father_node', father_node['name'], father_node['input'], father_node['output'])
        # print ('son_node', son_node['name'], son_node['input'], son_node['output'])

    @staticmethod
    def check_one_of_input_is_const(node, graph):
        """
         check one of input is const
        :param node:
        :param graph:
        :return:
        """
        for i in node['input']:
            if graph[i['name']]['type'] == 'Const':
                return True
        return False

    @staticmethod
    def _auto_split(med_node, med_graph):
        """
        add split to node which output size >=2
        :param med_node:
        :param med_graph:
        :return:
        """
        output = med_node['output']
        #print 'output!!:', output
        if len(output) > 1:
            split_node = MedNodeUtil.new_med_node()
            split_node['name'] = med_node['name'] + '_split#' + str(len(output))
            split_node['ak_type'] = 'Split'
            split_node['type'] = 'Split'
            split_node['ak_attr']['split_num'] = len(output)
            # print ('-------------')
            # print ('split', split_node['name'])
            MedGraphUtil.append_node(med_node, split_node, graph=med_graph)
        pass

    @staticmethod
    def _auto_input_name(med_node, med_graph):
        """
        gen input name
        :param med_node:
        :param med_graph:
        :return:
        """
        assert med_node['ak_type'] == 'Input'
        old_name = med_node['name']
        med_node['name'] = 'input_' + str(MedGraph_Input_Cnt)
        for i in med_node['output']:
            out_node = med_graph[i]
            out_node['input'] = MedNodeUtil.replace_name_with_list(out_node['input'], old_name,
                                                                   [[med_node['name']]])
    @staticmethod
    def _fusionPermute(med_node, med_graph):
        """
        when permute param >= 5, fusion Permute node to pixelshuffle
        :param med_node:
        :param med_graph:
        :return:
        """
        if len(med_node['ak_attr']['shape']) >= 5:
           ins = med_node['input']
           outs = med_node['output']
           if len(ins) == 1 and len(outs) == 1:
               in_node = med_graph[ins[0]]
               out_node = med_graph[outs[0]]
               if in_node['ak_type'] == 'Reshape' and out_node['ak_type'] == 'Reshape':
                #    print(in_node)
                   rw = in_node['ak_attr']['shape'][1]
                   rh = in_node['ak_attr']['shape'][2]
                   in_node['type'] = 'PixelShuffle'
                   in_node['ak_type'] = 'PixelShuffle'
                   in_node['ak_attr']['type'] = 'PixelShuffle'
                   in_node['ak_attr']['rw'] = int(rw)
                   in_node['ak_attr']['rh'] = int(rh)
                   in_node['ak_attr']['channel_first'] = True
                   #delete med_node and out_node
                   in_node['output']=out_node['output']
                   for i in out_node['output']:
                       in_node_node = med_graph[i]
                       in_node_node['input'] = MedNodeUtil.replace_name_with_list(in_node_node['input'],
                                                                   out_node['name'],
                                                                   [in_node['name']])
                    #    print(in_node_node['input'])
                #    print(in_node)
                   med_graph.pop(med_node['name'])
                   med_graph.pop(out_node['name'])

    @staticmethod
    def _fusionScale(med_node, med_graph):
        """
        fusion scale node after convolution node
        :param med_node:
        :param med_graph:
        :return:
        """
        if len(med_node['input']) == 1:
            input_node = med_graph[med_node['input'][0]]
            med_ak_attr = med_node['ak_attr']
            if input_node['ak_type'] == 'Convolution':
                input_attr = input_node['ak_attr']
                conv_weights = input_attr['weights']
                scale_weights = med_ak_attr['weights']

                assert (conv_weights['shape'][0] == scale_weights['shape'][-1]) \
                        or (conv_weights['shape'][0] == scale_weights['shape'][0])
                shape = conv_weights['shape']
                new_conv_weights = {}
                new_conv_weights['shape'] = conv_weights['shape']
                new_conv_weights['dtype'] = 'float32'
                new_conv_weights['data'] = np.zeros(shape)
                tmp = scale_weights['data'].flatten()
                conv_weights['data'] = conv_weights['data'].reshape(shape)
                for i in range(shape[0]):
                    new_conv_weights['data'][i] = conv_weights['data'][i] * tmp[i]
                input_attr['weights'] = new_conv_weights
                if input_attr.get('bias') is not None:
                    bias_val = input_attr['bias']
                    if 'bias' in med_ak_attr:
                        new_conv_bias = {}
                        new_conv_bias['shape'] = bias_val['shape']
                        new_conv_bias['dtype'] = 'float32'
                        new_conv_bias['data'] = np.zeros(bias_val['shape'])
                        med_val = med_ak_attr['bias']
                        for i in range(bias_val['shape'][0]):
                            new_conv_bias['data'][i] = bias_val['data'][i] + med_val['data'][i]
                        input_attr['bias'] = new_conv_bias
                    else:
                        input_attr['bias'] = bias_val
                elif med_ak_attr.get('bias') is not None:
                    bias_val = med_ak_attr['bias']
                    input_attr['bias'] = bias_val
                else:
                    print ('conv+scale does not have bias')
                    # input_attr['bias'] = med_ak_attr['bias']
                med_node['ak_type'] = None
                input_node['output'] = MedNodeUtil.replace_name_with_list(input_node['output'],
                                                                          med_node['name'],
                                                                          med_node['output'])
                MedNodeUtil.redirecto_outputs_input_to_this(med_node, med_graph, input_node['name'])
                input_node['fusion_out_name'] = med_node['name']
                # conv+scale+scale * n, bias_n1 = bias_n0 * weights + bias_n1
                if len(input_node['output']) == 1:
                    tmp_node = med_graph[input_node['output'][0]]
                    while tmp_node['ak_type'] == 'Scale':
                        input_attr = input_node['ak_attr']
                        conv_weights = input_attr['weights']
                        scale_weights = tmp_node['ak_attr']['weights']
                        assert (conv_weights['shape'][0] == scale_weights['shape'][-1]) or (conv_weights['shape'][0] == scale_weights['shape'][0])
                        shape = conv_weights['shape']
                        new_conv_weights = {}
                        new_conv_weights['shape'] = conv_weights['shape']
                        new_conv_weights['dtype'] = 'float32'
                        new_conv_weights['data'] = np.zeros(shape)
                        tmp = scale_weights['data'].flatten()
                        conv_weights['data'] = conv_weights['data'].reshape(shape)
                        for i in range(shape[0]):
                            new_conv_weights['data'][i] = conv_weights['data'][i] * tmp[i]
                        input_attr['weights'] = new_conv_weights
                        if input_attr.get('bias') is not None:
                            bias_val = input_attr['bias']
                            if 'bias' in tmp_node['ak_attr']:
                                new_conv_bias = {}
                                new_conv_bias['shape'] = bias_val['shape']
                                new_conv_bias['dtype'] = 'float32'
                                new_conv_bias['data'] = np.zeros(bias_val['shape'])
                                med_val = tmp_node['ak_attr']['bias']
                                for i in range(bias_val['shape'][0]):
                                    new_conv_bias['data'][i] = bias_val['data'][i] * scale_weights['data'][i] + med_val['data'][i]
                                input_attr['bias'] = new_conv_bias
                            else:
                                input_attr['bias'] = bias_val
                        elif med_ak_attr.get('bias') is not None:
                            bias_val = tmp_node['ak_attr']['bias']
                            input_attr['bias'] = bias_val
                        else:
                            print ('conv+scale does not have bias')
                        tmp_node['ak_type'] = None
                        input_node['output'] = MedNodeUtil.replace_name_with_list(input_node['output'],
                                                                          tmp_node['name'],
                                                                          tmp_node['output'])
                        MedNodeUtil.redirecto_outputs_input_to_this(tmp_node, med_graph, input_node['name'])
                        input_node['fusion_out_name'] = tmp_node['name']
                        if len(input_node['output']) == 1:
                            tmp_node = med_graph[input_node['output'][0]]
                        else:
                            break

        pass

    @staticmethod
    def _deleteScale(med_node, med_graph):
        """
        delete dropout node when is_test = 0
        :param med_node:
        :param med_graph:
        :return:
        """
        ak_attr = med_node['ak_attr']
        if 'drop' in ak_attr.keys() and ak_attr['drop'] == 0:
            #not do scale, delete node
            input = med_node['input']
            output = med_node['output']
            # print ('name: ', med_node['name'])
            # print ('inputs: ', input)
            # print ('outputs: ', output)
            #replace node
            for node in input:
                for out in med_graph.keys():
                    if out == node:
                        out_node = med_graph[out]['output']
                        # print 'name: ', out
                        # print 'input: ', med_graph[out]['input']
                        # print 'output: ', out_node
                        for i in range(0, len(out_node)):
                            if out_node[i] == med_node['name']:
                                out_node.pop(i)
                                out_node += output
                                # print 'name: ', out
                                # print 'input: ', med_graph[out]['input']
                                # print 'output: ', out_node
                                break
            for node in output:
                for out in med_graph.keys():
                    if out == node:
                        in_node = med_graph[out]['input']
                        # print 'name: ', out
                        # print 'input: ', in_node
                        # print 'output: ', med_graph[out]['output']
                        for i in range(0, len(in_node)):
                            if in_node[i] == med_node['name']:
                                in_node.pop(i)
                                in_node += input
                                # print 'name: ', out
                                # print 'input: ', in_node
                                # print 'output: ', med_graph[out]['output']
            # print ('pop: ', med_node['name'])
            med_graph.pop(med_node['name'])
            # print ('graph: -----')
            # for key in med_graph.keys():
            #     node = med_graph[key]
            #     print(node['name'], node['ak_type'], node['input'], node['output'])
            #del med_graph[med_node]
        pass

    @staticmethod
    def _all_search_table(graph, table):
        """
        search template for dict
        :param graph:
        :param table:
        :return:
        """
        for onnx_node in graph.values():
            if onnx_node['med_visted']:
                continue
            type_name = onnx_node['ak_type']
            if table.get(type_name) is not None:
                table[type_name](onnx_node, graph)

    @staticmethod
    def _all_search_fusion(graph, fusion_func):
        """
        search template for func
        :param graph:
        :param fusion_func:
        :return:
        """
        for onnx_node in graph.values():
            if onnx_node['med_visted']:
                continue
            if onnx_node['ak_type'] is not None:
                fusion_func(onnx_node, graph)

    @staticmethod
    def solve(med_graph):
        """
        do fusion and adjust for med graph that we can convert med graph to ak graph
        :param med_graph:
        :return:
        """
        for node in med_graph.values():
            node['med_visted'] = False

        print ('********split***********')
        MedGraphUtil._all_search_fusion(med_graph, MedGraphUtil._auto_split)
        print ('********scale***********')
        MedGraphUtil._all_search_table(med_graph, {'Scale': MedGraphUtil._deleteScale})
        print ('********pixelShuffle***********')
        MedGraphUtil._all_search_table(med_graph, {'Permute': MedGraphUtil._fusionPermute})
        print ('********fusion scale***********')
        MedGraphUtil._all_search_table(med_graph, {'Scale': MedGraphUtil._fusionScale})
        print ('********finish***********')
        # MedGraphUtil._all_search_table(med_graph, {'Input': MedGraphUtil._auto_input_name})

    @staticmethod
    def search_output_list(graph):
        """
        search output list in recursive method
        :param graph:
        :return:
        """
        output_list = set()
        graph_cp = graph.copy()

        def recursive_search(node):
            if node.get('out_search_flag') is not None:
                return set()
            node['out_search_flag'] = True
            outputs = node['output']
            result = set()
            if len(outputs) == 0:
                result.add(node['name'])
            else:
                for i in outputs:
                    result |= recursive_search(graph[i['name']])
            return result

        for i in graph_cp.values():
            output_list |= recursive_search(i)
        return list(output_list)
