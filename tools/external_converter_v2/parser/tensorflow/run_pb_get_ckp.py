import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import numpy as np
from tensorflow.python.platform import gfile

# graph_path = '/tmp/pycharm_project_635/external_converter_v2/parser/kill_tf/mnist_model/graph.pbtxt'
# graph_path = './resnet_v1_50_graph.pb'
# graph_path = './frozen_graph.pb'
# graph_path='./ease_model/model.cpkt.meta'
# graph_path='./ease_model/graph.pb'
# graph_path='./ease_model/frozen_mnist.pb'
# graph_path='./vgg_model/frozen_vgg_16_i.pb'
graph_path = './inception_model/inception_v2_inf_graph.pb'
cpkt_path = './inception_model/'
# graph_path='./resnet_model/frozen_resnet_v1_50.pb'


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
sess = tf.Session(graph=graph)
x = graph.get_tensor_by_name('graph/input:0')
y = graph.get_tensor_by_name('graph/InceptionV2/Predictions/Reshape_1:0')

for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(var.name)
init_table = tf.tables_initializer()
sess.run(init_table)
init = tf.global_variables_initializer()
sess.run(init)
out = sess.run(y, {x: np.ones((1, 224, 224, 3))})

saver = tf.train.Saver()
saver.save(sess, cpkt_path + '/model.cpkt')
