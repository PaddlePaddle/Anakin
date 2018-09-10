import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

# import tensorflow as tf
# with tf.Session() as sess:
#     with open('./resnet_v1_50_graph.pb', 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         print(graph_def)


from tensorflow.python.platform import gfile

# graph_path = './resnet_v1_50_graph.pb'
# graph_path = './frozen_graph.pb'
# graph_path='./vgg_model/frozen_vgg_16_i.pb'
# graph_path='./inception_model/frozen_inception_v2.pb'
graph_path = './resnet_model/frozen_resnet_v1_50.pb'


# graph_path='./mobilnetv2/frozen_mobilnet_v2.pb'

def get_graph(graph_path):
    '''
    get tensor board graph from pb file or meta file
    :param graph_path:
    :return:
    '''
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
    return graph


graph = get_graph(graph_path)
summaryWriter = tf.summary.FileWriter('log/', graph)
