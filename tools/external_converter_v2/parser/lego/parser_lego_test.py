import numpy as np
import os
from ..graph_io import *
from ..logger import *
from ..proto import *
from parser_util import *
from lego_layer_param_transmit import *
import pdb
from utils_lego import HandleBlob
from utils_lego import CreateEdges

class LegoParser_test:
    def __init__(self, lego_config_dict):
        # caffe net parameter
        self.net_parameter = NetParameter()
        self.net_param_weights = NetParameter()
        # anakin graph model io
        self.graphIO = GraphProtoIO()
        # config info
        self.ProtoPaths = lego_config_dict['ProtoPaths']
        self.PrototxtPath = lego_config_dict['PrototxtPath'] 
        self.ModelPath = lego_config_dict['ModelPath']
        self.data = HandleBlob()
        self.connet_edges = CreateEdges()
    def __call__(self):
        return self._Parsing()
    def _DetectionArch(self):
        self._ParserPrototxt()
        #self._FilterNet()
    def _ParserPrototxt(self):
        with open(self.PrototxtPath, "r") as f:
            text_format.Merge(f.read(), self.net_parameter)
    def _Parsing(self):
        print 'Parsing start...'
        self._DetectionArch()
        self.graphIO.set_name(self.net_parameter.name)
        source_inputs = self.net_parameter.input
        source_layers = self.net_parameter.layer
        self.data.deal_with_blob(source_layers)
        self.data.get_cur_btm_layer(source_layers)
        self.data.set_input_layer(source_inputs)
        with open(self.ModelPath, 'rb') as f:
            expect_model_size = os.path.getsize(self.ModelPath)
            sum_s = 0
            layer_cache = {}
            # deal with each layer
            for source_layer in source_layers:
                source_layer_name = source_layer.name
                source_layer_type = source_layer.type
                nodeIO = NodeProtoIO()
                opIO = OpsProtoIO()
                if source_layer_name not in layer_cache.keys():	# The FIRST appearance of layer
                    f.seek(sum_s)
                    layer_cache[source_layer_name] = [sum_s, 0]
                    node_name = source_layer_name
                else:
                    f.seek(layer_cache[source_layer_name][0])
                    layer_cache[source_layer_name][1] += 1		# The nodes of REPEATED layer need to add the suffix name
                    node_name = source_layer_name + '__multiplex__' + bytes(layer_cache[source_layer_name][1])
                nodeIO.set_name(node_name)
                opIO.set_name(source_layer_type)
                # create the connection of edges
                self.connet_edges.create(source_layer, node_name, self.data, nodeIO, self.graphIO)
                #get weights from lego.
                tensors = []
                size_list = blob_size_of_layer(source_layer)
                if len(size_list):
                    for size in size_list:
                        data = np.fromfile(f, '<f4', size)
                        tensor = TensorProtoIO()
                        tensor.set_data_type(FLOAT)
                        tensor.set_data(data, "float")
                        tensors.append(tensor)
                        if layer_cache[source_layer_name][1] == 0:	# The FIRST appearance of layer
                            sum_s = sum_s + size * 4
                        else:
                            pass
                # print source_layer
                LEGO_NODE_FILLER[source_layer_type](nodeIO, source_layer, tensors, opIO)   # Fill the nodeIO
                self.graphIO.add_node(nodeIO())    # Add the nodeIO
                #print nodeIO()
            if expect_model_size == sum_s:
                print "Correct size."
            else:
                print "Not Correct size"
        
        # delete some unnecessary edges.
        self.connet_edges.delete_edges(self.graphIO)
        #change edgs in bottoms's order.
        self.connet_edges.change_order(self.graphIO, self.data)
        return self.graphIO




