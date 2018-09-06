import tensorflow as tf
import numpy as np
import os
import tempfile
from parse_tf_2_med import ParseTF2Med
from med_graph import MedGraphUtil
from tf_trans_util import load_graph
from tf_util import TfUtil


class AutoDebug:

    def __init__(self, tf_model_path, ak_output_dir, workspace=None):
        self.ak_perfix = 'record_'
        self.tf_model_path = tf_model_path
        self.ak_output_dir = os.path.dirname(ak_output_dir)
        med_graph = ParseTF2Med(self.tf_model_path).parse()
        MedGraphUtil.solve(med_graph)
        self.med_graph = med_graph
        # if workspace is None:
        #     self.workspace=tempfile.mkdtemp()
        # else:
        #     self.workspace = workspace
        # print('workspace is ', self.workspace)

    def _debug_graph(self, graph):
        '''
        print debug info
        :param graph:
        :return:
        '''
        for i in graph.values():
            print(i)
        exit()

    def _convert_name_tf2ak(self, tf_name):
        '''
        conver name from tf 2 ak
        :param tf_name:
        :return:
        '''
        ak_name = tf_name[:]
        for index, x in enumerate(tf_name):
            if x == '/':
                ak_name = ak_name[:index] + '_' + ak_name[index + 1:]
        return self.ak_perfix + ak_name

    def _find_ak_output(self):
        '''
        find output name in graph
        :return:
        '''
        result = {}
        for file_name in os.listdir(self.ak_output_dir):
            real_path = os.path.join(self.ak_output_dir, file_name)
            if not os.path.isdir(real_path) and file_name.startswith(self.ak_perfix) and not file_name.startswith(
                    self.ak_perfix + '_'):
                result[file_name] = [float(i.split(' ')[1]) for i in open(real_path, 'r').readlines()]
        return result

    def _creat_and_clear_workspace(self):
        '''
        remove temp workspace
        :return:
        '''
        if os.path.exists(self.workspace):
            assert os.path.isdir(self.workspace)
            ls = os.listdir(self.workspace)
            for i in ls:
                c_path = os.path.join(self.workspace, i)
                os.remove(c_path)
        else:
            os.mkdir(self.workspace)

    def _prepare_data(self, med_graph):
        '''
        gen input data
        :param med_graph:
        :return:
        '''
        input_cnt = 0
        inputs = {}
        result = {}
        np.random.seed(1234567)
        for i in med_graph.values():
            if i['ak_type'] == 'Input':
                shape = i['output'][0]['shape']
                size = 1
                for k in shape:
                    size *= k
                tensor = np.random.randn(size).reshape(shape)
                inputs['input_' + str(input_cnt)] = tensor
                result[i['name'] + ':0'] = tensor
                input_cnt += 1

        for i in inputs.keys():
            import struct
            with open(os.path.join(self.ak_output_dir, i), "wb") as file:
                x_value_new = inputs[i].transpose((0, 3, 1, 2))
                for value in x_value_new.flatten():
                    file.write(struct.pack('f', value))

        return result

    def run_ak(self):
        '''
        run ak mode
        :return:
        '''
        self._prepare_data(self.med_graph)

    def run(self):
        '''
        run debug mode
        :return:
        '''
        pass
        # inputs = self._prepare_data(self.med_graph)
        # akopname2ak_out=self._find_ak_output()
        # for i in akopname2ak_out:
        #     print(i)
        # compare_dic={}
        # for node in self.med_graph.values():
        #     print(node['name'], node['ak_type'])
        #     if node['ak_type'] is not None and node['type'] is not None:
        #
        #         name=node['name']
        #         if node.get('fusion_out_name') is not None:
        #             name=node['fusion_out_name']
        #         ak_name=self._convert_name_tf2ak(name)
        #         print('key ',ak_name)
        #         if akopname2ak_out.get(ak_name) is not None:
        #             compare_dic[name+':0']=akopname2ak_out[ak_name]
        # outlist=compare_dic.keys()
        # out=TfUtil.tf_run_model(self.tf_model_path,inputs,outlist)
        # for index,name in enumerate(outlist):
        #     correct=out[index].transpose()
        #     ak_result=akopname2ak_out[name]
        #     assert len()
        #     for i in range(len(ak_result)):
        #         if abs()
        # tensor_shape = {}
        # graph={}
        # for node in tf_ops:
        #     for out in node.outputs:
        #         tensor_shape[out.name] = out.get_shape().as_list()
        #     input_names = [i.name for i in node.inputs]
        #     output_names = [i.name for i in node.outputs]
        #     graph[node.name]={'name':node.name,'input':input_names,'output':output_names}
        # self._fix_self_output(graph,tensor_shape)
        # self._debug_graph(graph)


ak_work_space = ''

debug = AutoDebug('./resnet_model/frozen_resnet_v1_50.pb', ak_work_space)
debug.run_ak()
debug.run()
