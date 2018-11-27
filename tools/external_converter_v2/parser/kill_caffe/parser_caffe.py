#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from ..graph_io import *
from ..logger import *
from ..proto import *
from caffe_helper import *
from caffe_layer_param_transmit import *
from Queue import Queue


class CaffeParser:
    """
    """
    def __init__(self, caffe_config_dict):
        """
        """
        # caffe net parameter
        self.net_parameter = NetParameter()
        self.net_param_weights = NetParameter()
        # anakin graph model io
        self.graphIO = GraphProtoIO()
        # config info
        self.ProtoPaths = caffe_config_dict['ProtoPaths']
        self.PrototxtPath = caffe_config_dict['PrototxtPath'] 
        self.ModelPath = caffe_config_dict['ModelPath']

    def __call__(self):
        """
        callable caffe parser
        """
        #return self._Parsing()
        return self._Parsing_new()

    def _DetectionArch(self):
        """
        """
        self._ParserPrototxt()
        self._UpgradeNetAsNeeded()
        self._FilterNet()
        self._InsertSplits()
        self._ScatterInputLayer()
        # create input node
        #self._CreateInputNode() maybe not need

    def _ParserPrototxt(self):
        """
        don't need to be used.
        """
        with open(self.PrototxtPath, "r") as f:
            text_format.Merge(f.read(), self.net_parameter)

    def _ParserModel(self):
        with open(self.ModelPath, "r") as f:
            self.net_param_weights.MergeFromString(f.read())

    def _UpgradeNetAsNeeded(self):
        """
        same as caffe UpgradeNetAsNeeded.
        """
        if NetNeedsV0ToV1Upgrade(self.net_parameter):
            # NetParameter was specified using the old style (V0LayerParameter), need to upgrade.
            logger(verbose.INFO).feed("[ Upgrade Level 1 ]  Details: need to upgrade from V0 to V1 [ ... ]")
            original_param = NetParameter()
            original_param.CopyFrom(self.net_parameter)
            if UpgradeV0Net(original_param, self.net_parameter):
                logger(verbose.WARNING).feed("[ Upgrade Level 1 ]  Details: need to upgrade from V0 to V1 [ SUC ]")
            else:
                logger(verbose.FATAL).feed("[ Upgrade Level 1 ]  Details: need to upgrade from V0 to V1 [ FAILED ]")
                exit()
        if NetNeedsDataUpgrade(self.net_parameter):
            logger(verbose.ERROR).feed("[ Upgrade Level 2 ] Details: need Data upgrade [ IGNORED ]")
        if NetNeedsV1ToV2Upgrade(self.net_parameter):
            logger(verbose.INFO).feed("[ Upgrade Level 3 ] Details: need to upgrade from V1 to V2 [ ... ]")
            original_param = NetParameter()
            original_param.CopyFrom(self.net_parameter)
            if UpgradeV1Net(original_param, self.net_parameter):
                logger(verbose.WARNING).feed("[ Upgrade Level 3 ] Details: need to upgrade from V1 to V2 [ SUC ]")
            else:
                logger(verbose.FATAL).feed("[ Upgrade Level 3 ] Details: need to upgrade from V1 to V2 [ FAILED ]")
                exit()
        if NetNeedsInputUpgrade(self.net_parameter):
            logger(verbose.INFO).feed("[ Upgrade Level 4 ] Details: need Input upgrade [ ... ]")	
            UpgradeNetInput(self.net_parameter)
            logger(verbose.WARNING).feed("[ Upgrade Level 4 ] Details: need Input upgrade [ SUC ]")
        if NetNeedsBatchNormUpgrade(self.net_parameter):
            logger(verbose.INFO).feed("[ Upgrade Level 5 ] Details: need BatchNorm upgrade [ ... ]")
            UpgradeNetBatchNorm(self.net_parameter)
            logger(verbose.INFO).feed("[ Upgrade Level 5 ] Details: need BatchNorm upgrade [ ... ]")

    def _InsertSplits(self):
        """
        Same as caffe InsertSplits.
        """
        param_split = NetParameter()
        layers = self.net_parameter.layer or self.net_parameter.layers

        # map: layer_idx  --> layer_name(string) 
        layer_idx_to_name = {}
        # map: blob_name --> (layer_idx(int), top_idx(int)), will be used soon after
        self.blob_name_to_last_top_idx = {}
        # map: (layer_idx(int), btm_idx(int)) --> (layer_idx(int), top_idx(int)) will be used soon after
        self.bottom_idx_to_source_top_idx = {}
        # map: (layer_idx(int), top_idx(int)) --> same as btm name (int)
        top_idx_to_bottom_count = {}
        # map:
        top_idx_to_bottom_split_idx = {}

        for idx, layer in enumerate(layers):
            #logger(verbose.INFO).feed(idx," layer name: ", layer.name, " type: ", layer.type)
            layer_idx_to_name[idx] = layer.name
            for j, btm in enumerate(layer.bottom):
                if btm not in self.blob_name_to_last_top_idx.keys():
                    logger(verbose.FATAL).feed("Unknown bottom (blob: %s) in (layer: '%s')" % (btm, layer.name))		
                    exit()
                bottom_idx = (idx, j)
                top_idx = self.blob_name_to_last_top_idx[btm]
                self.bottom_idx_to_source_top_idx[bottom_idx] = top_idx
                top_idx_to_bottom_count[top_idx] = top_idx_to_bottom_count[top_idx] + 1 if dict_has_key(top_idx_to_bottom_count, top_idx) else 1
            for j, top in enumerate(layer.top):
                self.blob_name_to_last_top_idx[top] = (idx, j)
        # add split layer 
        for idx, layer in enumerate(layers):
            layer_param = param_split.layer.add()
            layer_param.CopyFrom(layer)
            for j, btm in enumerate(layer_param.bottom):
                top_idx = self.bottom_idx_to_source_top_idx[(idx, j)]
                split_count = top_idx_to_bottom_count[top_idx]
                if split_count > 1:
                    layer_name = layer_idx_to_name[top_idx[0]]
                    blob_name = btm
                    if top_idx not in top_idx_to_bottom_split_idx:
                        top_idx_to_bottom_split_idx[top_idx] = 0
                    layer_param.bottom[j] = SplitBlobName(layer_name, blob_name, top_idx[1], top_idx_to_bottom_split_idx[top_idx])
                    top_idx_to_bottom_split_idx[top_idx] = top_idx_to_bottom_split_idx[top_idx] + 1
            for j, top in enumerate(layer_param.top):
                top_idx = (idx, j)
                if top_idx in top_idx_to_bottom_count:
                    split_count = top_idx_to_bottom_count[top_idx]
                else:
                    continue
                if split_count > 1:
                    layer_name = layer_idx_to_name[idx]
                    blob_name = top
                    split_layer_param = param_split.layer.add()
                    #ConfigureSplitLayer(layer_name, blob_name, j, split_count, loss_weight, split_layer_param)
                    # config split layer
                    split_layer_param.Clear()
                    split_layer_param.bottom.append(blob_name)
                    split_layer_param.name = SplitLayerName(layer_name, blob_name, j)
                    split_layer_param.type = "Split"
                    for k in range(split_count):
                        split_layer_param.top.append(SplitBlobName(layer_name, blob_name, j, k))
        # update
        if self.net_parameter.layer:
            del self.net_parameter.layer[:]
            self.net_parameter.layer.extend(param_split.layer)
        else:
            del self.net_parameter.layers[:]
            self.net_parameter.layers.extend(param_split.layer)

    def _ScatterInputLayer(self):
        """
        Scatter multi-input to single inputs layer because anakin needs to hold multi-input operations
        """ 
        scatter_net_input_layer = []
        layers = self.net_parameter.layers or self.net_parameter.layer
        for layer in layers:
            if layer.type == "Input":
                input_param = layer.input_param
                for idx, top in enumerate(layer.top):
                    tmp_input = LayerParameter()
                    tmp_input.name = "input_" + str(idx)
                    tmp_input.type = "Input"
                    tmp_input.top.append(top)
                    shape = tmp_input.input_param.shape.add()
                    shape.dim.extend(list(input_param.shape[idx].dim))
                    scatter_net_input_layer.append(tmp_input)
            else:
                scatter_net_input_layer.append(layer)
        if self.net_parameter.layers:
            del self.net_parameter.layers[:]
            self.net_parameter.layers.extend(scatter_net_input_layer)
        if self.net_parameter.layer:
            del self.net_parameter.layer[:]
            self.net_parameter.layer.extend(scatter_net_input_layer)

    def _FilterNet(self):
        """
        Filter out layers based on the current phase 'test'
        """
        layers = self.net_parameter.layers or self.net_parameter.layer
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        phase = "test"
        for idx, layer in enumerate(layers):
            layer_name = layer.name
            #logger(verbose.INFO).feed(" detect : [%s, %s] " % (layer_name, layer.type))
            if len(layer.include):
                phase = phase_map[layer.include[0].phase] 
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != 'test')
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == "Dropout")
            if not exclude:
                filtered_layers.append(layer)
                assert layer.name not in filtered_layer_names, " layer_name : %s" % (layer.name)
                filtered_layer_names.add(layer.name)
        if self.net_parameter.layers:
            del self.net_parameter.layers[:]
            self.net_parameter.layers.extend(filtered_layers)
        if self.net_parameter.layer:
            del self.net_parameter.layer[:]
            self.net_parameter.layer.extend(filtered_layers)

    def _CreateScaleOpForFaceUniqueBatchNorm(self, batchnorm_name):
        """
        this function only used for parsing Face caffe model
        """
        # create node scale
        scale_nodeIO = NodeProtoIO() # init a NodeProtoIO 
        scale_opIO = OpsProtoIO() # init a OpsProtoIO 
        scale_nodeIO.set_name(batchnorm_name + "_scale") # set node name 
        scale_opIO.set_name("Scale") # set op
        # change edge for graph_io
        ori_next_node_name_of_batchnorm = self.graphIO.get_edge_nexts(batchnorm_name)[0]
        self.graphIO.rm_edge(batchnorm_name, ori_next_node_name_of_batchnorm)
        self.graphIO.add_in_edge(batchnorm_name, batchnorm_name + "_scale")
        self.graphIO.add_out_edge(batchnorm_name, batchnorm_name + "_scale")
        self.graphIO.add_in_edge(batchnorm_name + "_scale", ori_next_node_name_of_batchnorm)
        self.graphIO.add_out_edge(batchnorm_name + "_scale", ori_next_node_name_of_batchnorm)
        # create layer scale
        scale_layer = LayerParameter()
        scale_layer.name = batchnorm_name + "_scale"
        scale_layer.type = "Scale"
        scale_layer.scale_param.bias_term = False
        return scale_nodeIO, scale_layer, scale_opIO

    def _CreateInputNode(self):
        """
        """
        node_io = NodeProtoIO()
        op_io = OpsProtoIO()
        inputs = list(self.net_parameter.input)
        if len(inputs):
            input_dim = map(int, list(self.net_parameter.input_dim))
            if not input_dim:
                if len(self.net_parameter.input_shape) > 0:
                    input_dim = map(int, self.net_parameter.input_shape[0].dim)
                    node_io.set_name(in_name)
                    node_io.add_in(in_name)
                    # leak out name , need to be added later.
                    shape = TensorShape()
                    shape.dim.value[:] = input_dim
                    shape.dim.size = len(input_dim)
                    node_io.add_attr("shape", shape, "shape")
                    op_io.set_name("Input")
                    op_io.set_in_num(1)
                    op_io.set_commutative(True)
                    node_io.set_op(op_io())
                    self.graphIO.add_node(node_io())
                    self.graphIO.add_in(in_name)
                else: 
                    # parser InputParameter instead.
                    logger(verbose.INFO).feed(" Need to parse the layer of type InputParameter.")
            else:
                for in_name in inputs:
                    node_io.set_name(in_name)
                    node_io.add_in(in_name)
                    # leak out name , need to be added later.
                    shape = TensorShape()
                    shape.dim.value[:] = input_dim
                    shape.dim.size = len(input_dim)
                    node_io.add_attr("shape", shape, "shape")
                    op_io.set_name("Input")
                    op_io.set_in_num(1)
                    op_io.set_commutative(True)
                    node_io.set_op(op_io())
                    self.graphIO.add_node(node_io())
                    self.graphIO.add_in(in_name)

    def _Parsing_new(self):
        """
        Parsering caffe model and caffe net file.
        Return:  GraphProto class
        """
        logger(verbose.INFO).feed(" [CAFFE] Parsing ...")
        self._DetectionArch()
        # get detected layer arch for inference
        real_layers = self.net_parameter.layers or self.net_parameter.layer
        # init base map info for detect edge in graph
        blob_btm_to_layer_name = {}
        blob_top_to_layer_name = {}
        for tmp_rlayer in real_layers:
            for btm in tmp_rlayer.bottom:
                if btm not in blob_btm_to_layer_name.keys():
                    blob_btm_to_layer_name[btm] = Queue(maxsize=0)
                    blob_btm_to_layer_name[btm].put(tmp_rlayer.name)
                else:
                    blob_btm_to_layer_name[btm].put(tmp_rlayer.name)
            for top in tmp_rlayer.top:
                if top not in blob_top_to_layer_name.keys():
                    blob_top_to_layer_name[top] = Queue(maxsize=0)
                    blob_top_to_layer_name[top].put(tmp_rlayer.name)
                else:
                    blob_top_to_layer_name[top].put(tmp_rlayer.name)
        # set graph proto's name
        self.graphIO.set_name(self.net_parameter.name)
        logger(verbose.ERROR).feed(" [CAFFE] Archtecture Parsing ...")

        # parsing model
        logger(verbose.ERROR).feed(" [CAFFE] Model Parameter Parsing ...")
        self._ParserModel()
        model_layers = self.net_param_weights.layers or self.net_param_weights.layer

        # we must setting graph edge first
        for idx, rlayer in enumerate(real_layers):
            source_layer_name = rlayer.name
            source_layer_type = rlayer.type
            # set link edge
            for top in rlayer.top:
                if top not in blob_btm_to_layer_name.keys():
                    self.graphIO.add_out(top + "_out", source_layer_name)
                    continue
                else:
                    if blob_btm_to_layer_name[top].empty():
                        self.graphIO.add_out(top + "_out", source_layer_name)
                        continue
                top_corr_btm_layer_name = blob_btm_to_layer_name[top].get()
                self.graphIO.add_out_edge(source_layer_name, top_corr_btm_layer_name)
                if source_layer_type == "Input":
                    self.graphIO.add_in(source_layer_name)
            for btm in rlayer.bottom:
                btm_corr_top_layer_name = blob_top_to_layer_name[btm].get()
                self.graphIO.add_in_edge(btm_corr_top_layer_name, source_layer_name)

        for idx, rlayer in enumerate(real_layers):
            source_layer_name = rlayer.name
            source_layer_type = rlayer.type
            logger(verbose.INFO).feed(" Dectect [%s:\t%s] " % (source_layer_type, source_layer_name))
            # construct the node_io and op_io
            nodeIO = NodeProtoIO() # init a NodeProtoIO
            opIO = OpsProtoIO() # init a OpsProtoIO
            nodeIO.set_name(source_layer_name) # set node name
            opIO.set_name(source_layer_type) # set op name 

            opIO.set_out_num(len(rlayer.top)) 
            opIO.set_in_num(len(rlayer.bottom))

            match_in_model_layer = False
            # find corresponding model layer
            for mlayer in model_layers:
                if rlayer.name == mlayer.name: # find
                    #assert source_layer_type == mlayer.type, " real layer type(%s) must be equal to that(%s) of model layer." % (source_layer_type, mlayer.type)
                    logger(verbose.INFO).feed("  `--[ Match ]Parsing [%s:\t%s] " % (source_layer_type, source_layer_name))

                    # fill node with blobs parameter, such as filter and weights
                    tensors = []
                    if mlayer.blobs:
                        for blob in mlayer.blobs:
                            if blob in mlayer.blobs:
                                tensor = TensorProtoIO()
                                if len(blob.shape.dim):
                                    n, c, h, w = map(int, [1] * (4 - len(blob.shape.dim)) + list(blob.shape.dim))
                                    if len(blob.shape.dim) == 1:
                                        c = w
                                        w = 1
                                else:
                                    n, c, h, w = blob.num, blob.channels, blob.height, blob.width
                                #data = np.array(blob.data, dtype=np.float32).reshape(n, c, h, w)
                                tensor.set_data_type(FLOAT) # default float
                                if source_layer_type == "Deconvolution": # deconv is different in caffe
                                    tensor.set_shape([c, n, h, w])
                                else:
                                    tensor.set_shape([n, c, h, w]) # set shape (n c h w)
                                tensor.set_data(blob.data, "float")
                                tensors.append(tensor)
                    # fill node with layerparameter, such as axis kernel_size... and tensors
                    if len(tensors) > 3 and source_layer_type == "BatchNorm": # this is for Face unique Batchnorm layer(batchnorm + scale)
                        scale_node_io, scale_layer, scale_op_io = self._CreateScaleOpForFaceUniqueBatchNorm(source_layer_name)
                        CAFFE_LAYER_PARSER["Scale"](scale_node_io, scale_layer, tensors[3:5], scale_op_io)
                        self.graphIO.add_node(scale_node_io())
                        CAFFE_LAYER_PARSER[source_layer_type](nodeIO, mlayer, tensors[0:3], opIO)
                    else:
                        # besides, set the name of opIO
                        CAFFE_LAYER_PARSER[source_layer_type](nodeIO, rlayer, tensors, opIO) # call parser automatically
                    match_in_model_layer = True
                    # TODO... over!
                else: # not find
                    pass
            if not match_in_model_layer:
                # fill node with layerparameter, such as axis kernel_size... but with [ ] tensors (empty)
                # besides, set the name of opIO
                CAFFE_LAYER_PARSER[source_layer_type](nodeIO, rlayer, [], opIO) # call parser automatically
            # add node to graph io
            self.graphIO.add_node(nodeIO())

        return self.graphIO

    def _Parsing(self):
        """
        Parsering caffe model and caffe net file.
        Return:  GraphProto class
        """
        logger(verbose.INFO).feed(" [CAFFE] Parsing ...")
        self._DetectionArch()
        # get detected layer arch for inference
        real_layers = self.net_parameter.layers or self.net_parameter.layer
        # init base map info for detect edge in graph
        blob_btm_to_layer_name = {}
        blob_top_to_layer_name = {}
        for tmp_rlayer in real_layers:
            for btm in tmp_rlayer.bottom:
                if btm not in blob_top_to_layer_name.keys():
                    blob_top_to_layer_name[btm] = Queue(maxsize=0)
                    blob_top_to_layer_name[btm].put(tmp_rlayer.name)
                else:
                    blob_top_to_layer_name[btm].put(tmp_rlayer.name)
                #blob_top_to_layer_name[btm] = tmp_rlayer.name
            for top in tmp_rlayer.top:
                if top not in blob_btm_to_layer_name.keys():
                    blob_btm_to_layer_name[top] = Queue(maxsize=0)
                    blob_btm_to_layer_name[top].put(tmp_rlayer.name)
                else:
                    blob_btm_to_layer_name[top].put(tmp_rlayer.name)
                #blob_btm_to_layer_name[top] = tmp_rlayer.name
        # set graph proto's name
        self.graphIO.set_name(self.net_parameter.name)
        logger(verbose.ERROR).feed(" [CAFFE] Archtecture Parsing ...")

        # parsing model
        logger(verbose.ERROR).feed(" [CAFFE] Model Parameter Parsing ...")
        self._ParserModel()
        model_layers = self.net_param_weights.layers or self.net_param_weights.layer
        for idx, rlayer in enumerate(real_layers):
            source_layer_name = rlayer.name
            source_layer_type = rlayer.type
            logger(verbose.INFO).feed(" Dectect [%s:\t%s] " % (source_layer_type, source_layer_name))
            # construct the node_io and op_io
            nodeIO = NodeProtoIO() # init a NodeProtoIO
            opIO = OpsProtoIO() # init a OpsProtoIO
            nodeIO.set_name(source_layer_name) # set node name
            opIO.set_name(source_layer_type) # set op name
            # set link edge
            for btm in rlayer.bottom:
                btm_layer_name = blob_btm_to_layer_name[btm].get()
                # ensure that the que not empty
                if blob_btm_to_layer_name[btm].empty():
                    blob_btm_to_layer_name[btm].put(btm_layer_name)
                nodeIO.add_in(btm_layer_name)
                self.graphIO.add_in_edge(btm_layer_name, source_layer_name)
            opIO.set_in_num(len(rlayer.bottom))
            for top in rlayer.top:
                if top not in blob_top_to_layer_name.keys():
                    # add output node
                    self.graphIO.add_out(top + "_out", source_layer_name)
                    continue
                top_layer_name = blob_top_to_layer_name[top].get()
                if blob_top_to_layer_name[top].empty():
                    blob_top_to_layer_name[top].put(top_layer_name)
                nodeIO.add_out(top_layer_name)
                self.graphIO.add_out_edge(source_layer_name, top_layer_name)
                # add input node
                if source_layer_type == "Input":
                    self.graphIO.add_in(source_layer_name)
            opIO.set_out_num(len(rlayer.top))

            match_in_model_layer = False
            # find corresponding model layer
            for mlayer in model_layers:
                if rlayer.name == mlayer.name: # find
                    #assert source_layer_type == mlayer.type, " real layer type(%s) must be equal to that(%s) of model layer." % (source_layer_type, mlayer.type)
                    logger(verbose.INFO).feed("  `--[ Match ]Parsing [%s:\t%s] " % (source_layer_type, source_layer_name))

                    # fill node with blobs parameter, such as filter and weights
                    tensors = []
                    if mlayer.blobs:
                        for blob in mlayer.blobs:
                            if blob in mlayer.blobs:
                                tensor = TensorProtoIO()
                                if len(blob.shape.dim):
                                    n, c, h, w = map(int, [1] * (4 - len(blob.shape.dim)) + list(blob.shape.dim))
                                    if len(blob.shape.dim) == 1:
                                        c = w
                                        w = 1
                                else:
                                    n, c, h, w = blob.num, blob.channels, blob.height, blob.width
                                #data = np.array(blob.data, dtype=np.float32).reshape(n, c, h, w)
                                tensor.set_data_type(FLOAT) # default float
                                if source_layer_type == "Deconvolution": # deconv is different in caffe
                                    tensor.set_shape([c, n, h, w])
                                else:
                                    tensor.set_shape([n, c, h, w]) # set shape (n c h w)
                                tensor.set_data(blob.data, "float")
                                tensors.append(tensor)
                    # fill node with layerparameter, such as axis kernel_size... and tensors
                    if len(tensors) > 3 and source_layer_type == "BatchNorm": # this is for Face unique Batchnorm layer(batchnorm + scale)
                        scale_node_io, scale_layer, scale_op_io = self._CreateScaleOpForFaceUniqueBatchNorm(source_layer_name)
                        CAFFE_LAYER_PARSER["Scale"](scale_node_io, scale_layer, tensors[3:5], scale_op_io)
                        self.graphIO.add_node(scale_node_io())
                        CAFFE_LAYER_PARSER[source_layer_type](nodeIO, mlayer, tensors[0:3], opIO)
                    else:
                        # besides, set the name of opIO
                        CAFFE_LAYER_PARSER[source_layer_type](nodeIO, rlayer, tensors, opIO) # call parser automatically
                    match_in_model_layer = True
                    # TODO... over!
                else: # not find
                    pass
            if not match_in_model_layer:
                # fill node with layerparameter, such as axis kernel_size... but with [ ] tensors (empty)
                # besides, set the name of opIO
                CAFFE_LAYER_PARSER[source_layer_type](nodeIO, rlayer, [], opIO) # call parser automatically
            # add node to graph io
            self.graphIO.add_node(nodeIO())

        return self.graphIO
