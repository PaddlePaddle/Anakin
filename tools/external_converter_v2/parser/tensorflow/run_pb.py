import numpy as np
from tf_util import TfUtil


def convert_name_tf2ak(tf_name, perfix='record_'):
    '''
    conver tf name to ak name
    :param tf_name:
    :param perfix:
    :return:
    '''
    ak_name = tf_name[:]
    for index, x in enumerate(tf_name):
        if x == '/':
            ak_name = ak_name[:index] + '_' + ak_name[index + 1:]
    return perfix + ak_name


ak_work_space='/home/ljj/docker_mount_dev2/anakin2_developing/build'
output_compare_op = None

# graph_path='./vgg_model/frozen_vgg_16_i.pb'
# input_name='graph/input:0'
# output_op='graph/vgg_16/fc8/BiasAdd'
# output_compare_op='graph/vgg_16/fc8/convolution'

graph_path = './resnet_model/frozen_resnet_v1_50.pb'
input_name = 'graph/input:0'
output_op = 'graph/resnet_v1_50/predictions/Reshape_1'

# graph_path='./mobilnetv2/frozen_mobilnet_v2.pb'
# input_name='graph/input:0'
# output_op='graph/MobilenetV2/Predictions/Reshape_1'

# graph_path='./inception_model/frozen_inception_v2.pb'
# input_name='graph/input:0'
# output_op='graph/InceptionV2/Predictions/Reshape_1'


is_compared = True
output_name = output_op + ':0'
if output_compare_op is None:
    compare_path = ak_work_space + convert_name_tf2ak(output_op)
else:
    compare_path = ak_work_space + convert_name_tf2ak(output_compare_op)

np.random.seed(1234567)
x_location = ak_work_space + 'input_x'
x_value = np.random.randn(1, 224, 224, 3)
outs = TfUtil.tf_run_model(graph_path, {input_name: x_value}, [output_name])

import struct

with open(x_location, "wb") as file:
    x_value_new = x_value.transpose((0, 3, 1, 2))
    for value in x_value_new.flatten():
        file.write(struct.pack('f', value))

if is_compared:
    out = outs[0]
    if len(out.shape) == 4:
        out = out.transpose((0, 3, 1, 2))
    else:
        print('out shape :', out.shape)
    print(out.flatten()[:10])
    try:
        with open(compare_path, 'r')as file:
            ans = []
            for value_str in file.readlines():
                ans.append(float(value_str.split(' ')[1]))
            correct = out.flatten()
            assert len(correct) == len(ans)
            for i in range(len(ans)):
                if abs(ans[i] - correct[i]) > 0.0001 and abs(ans[i] - correct[i]) / abs(correct[i]) > 0.0001:
                    print(i, '=', ans[i], 'correct = ', correct[i])
                    exit()
        print('passed')
    except Exception, e:
        print(out)
        print('can`t find file : ' + compare_path)
