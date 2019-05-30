import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import numpy as np


class TfUtil:
    @staticmethod
    def tf_run_model(graph_path, inputs, output_tensor_list):
        '''
        fill inputs, run tensorflow mode and fetch output in output_tensor_list
        :param graph_path:
        :param inputs:
        :param output_tensor_list:
        :return:
        '''
        sess = tf.Session()
        if graph_path.endswith('.pbtxt'):
            input_binary = False
        else:
            input_binary = True

        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        if graph_path.endswith('.pb') or graph_path.endswith('.pbtxt'):
            mode = "rb" if input_binary else "r"
            with gfile.FastGFile(graph_path, mode) as f:
                if input_binary:
                    graph_def.ParseFromString(f.read())
                else:
                    text_format.Merge(f.read(), graph_def)
        else:
            tf.train.import_meta_graph(graph_path, clear_devices=True)

        tf.import_graph_def(graph_def, name='graph')
        for op in graph.get_operations():
            print(op.name, [i for i in op.inputs])
        inputs_dict = {graph.get_tensor_by_name(i): inputs[i] for i in inputs}
        output_list = [graph.get_tensor_by_name(i) for i in output_tensor_list]
        print(output_list)
        out = sess.run(output_list, inputs_dict)
        return out
