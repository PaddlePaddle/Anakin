import numpy as np
from ..graph_io import TensorProtoIO
from ..graph_io import NodeProtoIO
from ..graph_io import OpsProtoIO
from ..graph_io import GraphProtoIO
from lego_layer_param_transmit import *

class HandleBlob:
    def __init__(self):
        # btm ---> rename btm.
        self.btm2rename = dict()
        # top ---> rename top.
        self.top2rename = dict()
        # bottom layer of blob.
        self.btm_layer_of_blob = dict()
        # top layer of blob.
        self.top_layer_of_blob = dict()
        # layer_name
        self.layer_name = dict()
        # activation layer.
        self.activation_layer = dict()
        # the curent layer's bottom layer.
        self.btmLayerOfCur = dict()
        # input layers.
        self.input_layers_dic = {}

    def set_input_layer(self, source_inputs):
        for input_layer in source_inputs:
            name = input_layer.name
            name = name + "_input"
            self.input_layers_dic[name] = input_layer

    def get_cur_btm_layer(self, source_layers):
        for source_layer in source_layers:
            if self.layer_name[source_layer.name] == 0:
                # s = []
                if len(source_layer.bottoms) > 1:
                    self.btmLayerOfCur[source_layer.name] = []
                    for it in source_layer.bottoms:
                        if it in self.top2rename.keys():
                            it = self.top2rename[it]
                        self.btmLayerOfCur[source_layer.name].append(self.btm_layer_of_blob[it][0])
            else:
                # s = []
                if len(source_layer.bottoms) > 1:
                    self.btmLayerOfCur[source_layer.name+'__multiplex__' + bytes(self.layer_name[source_layer.name])] = []
                    for it in source_layer.bottoms:
                        if it in self.top2rename.keys():
                            it = self.top2rename[it]
                        self.btmLayerOfCur[source_layer.name+'__multiplex__' + bytes(self.layer_name[source_layer.name])].append(self.btm_layer_of_blob[it][0])

    def deal_with_blob(self, source_layers):
        # activation layers has the same tops and bottoms.
        # activation_layer = {}
        for source_layer in source_layers:
            tmp_tops = {}
            for top in source_layer.tops:
                tmp_tops[top] = True
            for btm in source_layer.bottoms:
                if btm in tmp_tops.keys():
                    self.activation_layer[source_layer.name] = btm
        
        # btm_layer_of_blob = {} # stored the bottom layers of blob.
        # top_layer_of_blob = {} # stored the top layers of blob.
        # layer_name = {}
        layer_index = {}
        index = 0
        for source_layer in source_layers:
            layer_index[source_layer.name] = index
            index += 1
        for source_layer in source_layers:
            # if source_layer.name == "VarSizeConv_0":
            # 	pdb.set_trace()
            # layer_name is to solve the situation that layer's name are the same.
            if source_layer.name not in self.layer_name.keys():
                self.layer_name[source_layer.name] = 0
            else:
                self.layer_name[source_layer.name] += 1
            for btm in source_layer.bottoms: # handle bottoms
                if source_layer.name in self.activation_layer.keys():
                    tmp = btm
                    btm = btm + '__' + source_layer.name + "'s_btm"
                    self.btm2rename[tmp] = btm
                elif btm in self.activation_layer.values():
                    tmp = [key for key in self.activation_layer.keys() if self.activation_layer[key]==btm]
                    tmp_str =""
                    tmp_ = btm
                    if layer_index[source_layer.name] > layer_index[tmp[0]]:
                        tmp_str = "'s_top"
                    # print '--------------tmp-str int top:---------------:'+tmp_str
                    btm = btm + '__' + tmp[0] + tmp_str
                    self.top2rename[tmp_] = btm
                if self.layer_name[source_layer.name] == 0:
                    # figure out some blobs has muti-tops and multi-bottoms.	
                    if btm not in self.top_layer_of_blob.keys():
                        self.top_layer_of_blob[btm] = [source_layer.name]
                    else:
                        self.top_layer_of_blob[btm].append(source_layer.name)
                else:
                    if btm not in self.top_layer_of_blob.keys():
                        self.top_layer_of_blob[btm] = [source_layer.name + '__multiplex__' + bytes(self.layer_name[source_layer.name])]
                    else:
                        self.top_layer_of_blob[btm].append(source_layer.name + '__multiplex__' + bytes(self.layer_name[source_layer.name]))

            for top in source_layer.tops: # handle top
                if source_layer.name in self.activation_layer.keys():
                    tmp = top
                    top = top + '__' + source_layer.name + "'s_top"
                    self.top2rename[tmp] = top
                elif top in self.activation_layer.values():
                    tmp = [key for key in self.activation_layer.keys() if self.activation_layer[key]==top]
                    tmp_ = top
                    tmp_str=""
                    if layer_index[source_layer.name] < layer_index[tmp[0]]:
                        tmp_str="'s_btm"
                    # print '--------------tmp-str int top:---------------:'+tmp_str
                    top = top + '__' + tmp[0] + tmp_str
                    self.btm2rename[tmp_] = top
                if self.layer_name[source_layer.name] == 0:
                    if top not in self.btm_layer_of_blob:
                        self.btm_layer_of_blob[top] = [source_layer.name]
                    else:
                        self.btm_layer_of_blob[top].append(source_layer.name)
                else:
                    if top not in self.btm_layer_of_blob:
                        self.btm_layer_of_blob[top] = [source_layer.name + '__multiplex__' + bytes(self.layer_name[source_layer.name])]
                    else:
                        self.btm_layer_of_blob[top].append(source_layer.name + '__multiplex__' + bytes(self.layer_name[source_layer.name]))

class CreateEdges:
    def __init__(self):
        self.btm_is_added = {} #avoid input node is to be re-added again.
        self.need_add_unpadding_layer = False
        self.unpadding_layer_id = 0
        self.split_id = 0
    
    def extern_add_in(self, input_node_name, out_node_name, source_layer, graphIO = GraphProtoIO()):
        """
        add graph input nodes.
        """
        tensors = []
        nodeIO = NodeProtoIO()
        nodeIO.set_name(input_node_name)
        nodeIO.add_out(out_node_name)
        opIO = OpsProtoIO()
        opIO.set_name("Input")
        nodeIO.set_op(opIO())
        graphIO.add_out_edge(input_node_name, out_node_name)
        graphIO.add_in_edge(input_node_name, out_node_name)
        LEGO_NODE_FILLER["Input"](nodeIO, source_layer, tensors, opIO)
        graphIO.add_node(nodeIO())
        graphIO.graph_proto.ins.append(input_node_name)

    def _AddSplit(self, source_layer_name, blob, tops_of_blob, split_id, need_add_upadding_layer, unpadding_layer_id, 
    split_num, graphIO=GraphProtoIO(), data = HandleBlob()):
        node = NodeProtoIO()
        node_name = "Split_"+str(split_id)
        node.set_name(node_name)
        # this is a specify code, and it is not for common use.
        # if need_add_upadding_layer==true, there is add unpadding layer into it.
        if need_add_upadding_layer:
            node1 = NodeProtoIO()
            node1_name = "Unpadding_padding_"+str(unpadding_layer_id)
            node1.set_name(node1_name)
            node1.add_in(source_layer_name)
            node1.add_out(node_name)
            # source_layer --> unpadding_padding layer
            graphIO.add_in_edge(source_layer_name, node1_name)
            graphIO.add_out_edge(source_layer_name, node1_name)
            # unpadding_padding layer ---> split
            graphIO.add_out_edge(node1_name, node_name)
            graphIO.add_in_edge(node1_name, node_name)
            # add op
            tensors1 = []
            op1 = OpsProtoIO()
            layer_type1 = "UnpaddingPaddingLayer"
            op1.set_name(layer_type1)
            LEGO_NODE_FILLER[layer_type1](node1, node1, tensors1, op1)
            graphIO.add_node(node1())
            # add in 
            node.add_in(node1_name)
        else:
            # add graph edges / source_layer--->split
            graphIO.add_in_edge(source_layer_name, node_name)
            graphIO.add_out_edge(source_layer_name, node_name)
            # add in
            node.add_in(source_layer_name)
        # change btm layer of cur layer.
        for top_layer in tops_of_blob[blob]:
            if top_layer in data.btmLayerOfCur.keys():
                id__ = 0
                for i in range(len(data.btmLayerOfCur[top_layer])):
                    if source_layer_name == data.btmLayerOfCur[top_layer][i]:
                        id__ = i
                data.btmLayerOfCur[top_layer][id__] = node_name
        for top_layer in tops_of_blob[blob]:
            node.add_out(top_layer)
            # add graph edges
            graphIO.add_out_edge(node_name, top_layer)
            graphIO.add_in_edge(node_name, top_layer)
        # add op
        tensors = []
        op = OpsProtoIO()
        layer_type = "BatchSplitLayer"
        op.set_name(layer_type)
        LEGO_NODE_FILLER[layer_type](node, split_num, tensors, op)				
        # add node to graph.
        graphIO.add_node(node())

    def create(self, source_layer, node_name, data = HandleBlob(), nodeIO = NodeProtoIO(), graphIO = GraphProtoIO()):
        source_layer_name = source_layer.name
        for btm in source_layer.bottoms:
            # get rename (RELU for example)
            if source_layer_name in data.activation_layer.keys():
                btm = data.btm2rename[btm]
            elif btm in data.top2rename.keys():
                btm = data.top2rename[btm]
            if btm not in data.btm_layer_of_blob.keys(): #input of graph
                if btm not in self.btm_is_added.keys():
                    self.btm_is_added[btm] = True
                    self.extern_add_in(btm+"_input", node_name, data.input_layers_dic[btm+"_input"], graphIO)
                else:
                    graphIO.add_out_edge(btm+"_input", node_name)
                    graphIO.add_in_edge(btm+"_input", node_name)
            else:
                for item in data.btm_layer_of_blob[btm]:
                    btm_layer_of_btm = item
                    nodeIO.add_in(btm_layer_of_btm)
                    graphIO.add_in_edge(btm_layer_of_btm, node_name)
        #opIO.set_in_num(len(source_layer.bottoms))
        # add a special layer, this is not for common use.
        if source_layer_name == "RELU_4" or source_layer_name == "RELU_6":
            self.need_add_unpadding_layer = True
            self.unpadding_layer_id += 1
        else:
            self.need_add_unpadding_layer = False

        for top in source_layer.tops:
            if source_layer_name in data.activation_layer.keys():
                top = data.top2rename[top]
            elif top in data.btm2rename.keys():
                top = data.btm2rename[top]
            if top not in data.top_layer_of_blob.keys(): # output of graph
                graphIO.add_out(top+"_out", node_name)
            else:
                # add split
                if len(data.top_layer_of_blob[top]) > 1:
                    split_num = len(data.top_layer_of_blob[top])
                    self._AddSplit(node_name, top, data.top_layer_of_blob, self.split_id, self.need_add_unpadding_layer, self.unpadding_layer_id, split_num, 
                    graphIO, data)
                    self.split_id += 1
                else:
                    top_layer_of_top = data.top_layer_of_blob[top]
                    nodeIO.add_out(top_layer_of_top[0])
                    graphIO.add_out_edge(node_name, top_layer_of_top[0])

    def delete_edges(self, graphIO = GraphProtoIO()):
        in_set = set()
        out_set = set()
        for i in graphIO.graph_proto.edges_in:
            # print "i:", i
            for j in graphIO.graph_proto.edges_in[i].val:
                in_set.add((j,i))
        for i in graphIO.graph_proto.edges_out:
            for j in graphIO.graph_proto.edges_out[i].val:
                out_set.add((i,j))
        ab_set = in_set - out_set
        ba_set = out_set - in_set
        if len(ab_set) >= 1:
            for it in ab_set:
                graphIO.rm_edge(it[0], it[1])
        if len(ba_set) >= 1:
            for it in ba_set:
                graphIO.rm_edge(it[0], it[1])

    # change some layer's inputs order.
    def change_order(self, graphIO = GraphProtoIO(), data = HandleBlob()):
        for i in graphIO.graph_proto.edges_in:
            s = []
            if i in data.btmLayerOfCur.keys():
                s = data.btmLayerOfCur[i]
                for j in range(len(data.btmLayerOfCur[i])):
                    graphIO.graph_proto.edges_in[i].val[j] = s[j]
