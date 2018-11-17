import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
from tensorflow.python.platform import gfile
from google.protobuf import text_format

graph_path = './inception_model/inception_v2_inf_graph.pb'
output_node_names = 'InceptionV2/Predictions/Reshape_1'
output_graph = './inception_model/inception_v2_inf_graph_empty.pb'

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

output_graph_def = graph_util.convert_variables_to_constants(
    sess,
    graph_def,
    output_node_names.split(",")  # We split on comma for convenience
)
# Finally we serialize and dump the output graph to the filesystem
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print("%d ops in the final graph." % len(output_graph_def.node))
