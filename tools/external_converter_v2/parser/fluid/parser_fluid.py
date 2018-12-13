import numpy as np
import paddle.fluid as fluid
import os
from ..graph_io import *
from ..logger import *
from ..proto import *
from fluid_layer_param_transmit import *

class FluidParser:

    def __init__(self, fluid_config_dict):
        # anakin graph model io
        self.graphIO = None
        # config info
        self.ModelPath = fluid_config_dict['ModelPath']
        self.NetType = fluid_config_dict['NetType']
        self.Debug = fluid_config_dict['Debug']
        # config fluid
        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.scope = fluid.core.Scope()
        # in and out edges of node
        self.ins = dict()
        self.outs = dict()
        # inplaced main node
        self.inplace_nodes = dict()
        self.graph_ins = list()
        self.graph_outs = list()
        self.scale_dict = dict()

    def __call__(self):
        return self._Parsing()

    def _NameNodeMid(self, op):
        first_outparam = op.output_names[0]
        arg_name = str(op.output(first_outparam)[0]).split('.')[0]
        #new_name = op.type + '_' + bytes(op.idx)
        new_name = op.type + '#' + bytes(op.idx) + '(' + arg_name + ')'
        return new_name

    def _NameNodeIn(self, in_name):
        new_name = 'input_' + bytes(self.graph_ins.index(in_name))
        return new_name

    def _NameNodeOut(self, out_name):
        new_name = out_name + '_gout'
        return new_name

    def _AddPairEdges(self, start_node_name, end_node_name, out_param, in_param):
        self.outs[start_node_name].add(out_param, end_node_name)
        self.ins[end_node_name].add(in_param, start_node_name)

    def _RmPairEdges(self, start_node_name, end_node_name):
        self.outs[start_node_name].rm(end_node_name)
        self.ins[end_node_name].rm(start_node_name)

    def _InitEdges(self, node_name):
        self.ins[node_name] = Fluid_edger()
        self.outs[node_name] = Fluid_edger()

    def _ClearEdges(self, node_name):
        if node_name.startswith('input_') is False:
            del self.ins[node_name]
        if node_name.endswith('_gout') is False:
            del self.outs[node_name]

    def _GetOp(self, ops, mid_node_name):
        mid_op = None
        for op in ops:
            node_name = self._NameNodeMid(op)
            if mid_node_name == node_name:
                mid_op = op
        return mid_op

    def _OpTypes(self, ops):
        types_cache = []
        for op in ops:
            if op.type not in types_cache:
                types_cache.append(op.type)
        return types_cache

    def _AddProtoNode(self, node_name, op_of_node, helper, private_data, op_type=None):
        nodeIO = NodeProtoIO()
        opIO = OpsProtoIO()
        nodeIO.set_name(node_name)
        if op_type is None:
            op_type = op_of_node.type
        FLUID_NODE_FILLER[op_type](nodeIO, op_of_node, opIO, helper, private_data)
        self.graphIO.add_node(nodeIO())

    def _RmProtoNode(self, node_name):
        self.graphIO.rm_node(self.graphIO.find_node_proto(node_name))

    def _InplaceNodes(self, position):
        inplace_heads = self.inplace_nodes.keys()
        inplace_mids = []
        inplace_ends = []
        for main_node_name in self.inplace_nodes.keys():
            mid_nodes_name = self.inplace_nodes[main_node_name][1: -1]
            inplace_mids.extend(mid_nodes_name)
        for main_node_name in self.inplace_nodes.keys():
            end_node_name = self.inplace_nodes[main_node_name][-1]
            inplace_ends.append(end_node_name)
        if position == 'Head':
            return inplace_heads
        elif position == 'Mid':
            return inplace_mids
        elif position == 'End':
            return inplace_ends
        elif position == 'All':
            return inplace_heads + inplace_mids + inplace_ends

    def _EdgeInplace(self, source_ops, helper):
        for source_op in source_ops:
            if source_op.type not in ['feed', 'fetch']:
                if len(source_op.input_arg_names) == 1 \
                and source_op.input_arg_names == source_op.output_arg_names:
                    source_node_name = self._NameNodeMid(source_op)
                    inplace_arg = source_op.input_arg_names[0]
                    for tmp_op in source_ops:
                        if tmp_op.idx != source_op.idx and inplace_arg in tmp_op.output_arg_names:
                            main_node_name = self._NameNodeMid(tmp_op)
                            if main_node_name not in self.inplace_nodes.keys():
                                self.inplace_nodes[main_node_name] = [main_node_name]
                            self.inplace_nodes[main_node_name].append(source_node_name)
        for main_node_name in self.inplace_nodes.keys():
            inplace_list = self.inplace_nodes[main_node_name]
            for inplace_node in inplace_list:
                idx = inplace_list.index(inplace_node)
                if idx != 0:
                    self.ins[inplace_node] = Fluid_edger('_In', inplace_list[idx - 1])
                if idx != len(inplace_list) - 1:
                    self.outs[inplace_node] = Fluid_edger('_Out', inplace_list[idx + 1])

    def _GetDebugOuts(self, source_ops, helper):
        if self.Debug == 'DEBUG':
            debug_fetch_list = []
            for source_op in source_ops:
                if source_op.type == 'fetch':
                    var_name = source_op.input_arg_names[0]
                    for tmp_op in source_ops:
                        if tmp_op.idx != source_op.idx and var_name in tmp_op.input_arg_names:
                            if var_name not in debug_fetch_list:
                                debug_fetch_list.append(var_name)
                        elif tmp_op.type == 'gru' and var_name in tmp_op.output_arg_names:
                            if var_name not in debug_fetch_list:
                                debug_fetch_list.append(var_name)
                        else:
                            pass
            return debug_fetch_list
        else:
            return []

    def _ParseBase(self, source_ops, helper, sub_graph_nodes=None):
        # Create the original base graph as described in fluid program.
        if sub_graph_nodes is None:
            sub_graph_nodes = list()
        self.graphIO = GraphProtoIO()
        self.graphIO.set_name('default_graph_name')
        debug_fetch_list = self._GetDebugOuts(source_ops, helper)
        self._EdgeInplace(source_ops, helper)
        for source_op in source_ops:
            if source_op.type not in ['feed', 'fetch']:
                main_node_name = self._NameNodeMid(source_op)
                in_edges = Fluid_edger()
                out_edges = Fluid_edger()
                for param in source_op.input_names:
                    if param not in ['InScale']:
                        for idx in range(0, len(helper.args_by_input_param(source_op, param))):
                            arg = helper.var_name_by_param(source_op, param, idx)
                            for tmp_op in source_ops:
                                if tmp_op.idx != source_op.idx and arg in tmp_op.output_arg_names:
                                    if tmp_op.type == 'feed':
                                        if arg not in self.graph_ins:
                                            self.graph_ins.append(arg)
                                            self.graphIO.add_in(self._NameNodeIn(arg))
                                        in_edges.add(param, self._NameNodeIn(arg), arg)
                                    else:
                                        tmp_node_name = self._NameNodeMid(tmp_op)
                                        if tmp_node_name in self.inplace_nodes.keys():
                                            inplace_node_name = self.inplace_nodes[tmp_node_name][-1]
                                            in_edges.add(param, inplace_node_name, arg)
                                        elif tmp_node_name not in self._InplaceNodes('All'):
                                            in_edges.add(param, tmp_node_name, arg)
                for param in source_op.output_names:
                    if param not in ['OutScale']:
                        for idx in range(0, len(helper.args_by_output_param(source_op, param))):
                            arg = helper.var_name_by_param(source_op, param, idx)
                            for tmp_op in source_ops:
                                if tmp_op.idx != source_op.idx and arg in tmp_op.input_arg_names:
                                    if tmp_op.type == 'fetch':
                                        if arg not in debug_fetch_list:
                                            arg_node_name = self._NameNodeOut(arg)
                                            if arg not in self.graph_outs:
                                                self.graph_outs.append(arg)
                                                self.graphIO.add_out_fluid(arg_node_name, \
                                                    main_node_name)
                                            out_edges.add(param, arg_node_name, arg)
                                            self.ins[arg_node_name] = Fluid_edger(bytes(source_op.idx), \
                                                main_node_name)
                                    else:
                                        out_edges.add(param, self._NameNodeMid(tmp_op), arg)
                self._AddProtoNode(main_node_name, source_op, helper, {})
                if main_node_name not in self._InplaceNodes('Mid'):
                    if main_node_name not in self._InplaceNodes('End'):
                        self.ins[main_node_name] = in_edges
                    if main_node_name not in self._InplaceNodes('Head'):
                        if main_node_name not in self._InplaceNodes('End'):
                            self.outs[main_node_name] = out_edges
                    else:
                        inplace_node_name = self.inplace_nodes[main_node_name][-1]
                        self.outs[inplace_node_name] = out_edges
                        for redundant_target in self.inplace_nodes[main_node_name][1:]:
                            self.outs[inplace_node_name].rm(redundant_target)

    def _PrintEdge(self, node, target, direction):
        var_name = 'Unknown'
        if direction == 'in':
            var = self.ins[node].vars_by_target(target)
        elif direction == 'out':
            var = self.outs[node].vars_by_target(target)
        if len(var) > 0:
            var_name = var[0]
        print node + ",\t" + target + ",\t" + var_name

    def _Graph(self, reverse=False, need_print=False):
        for node in self.ins.keys():
            targets_list = self.ins[node]()
            targets_scale = self.ins[node].all_scales()
            for idx, target in enumerate(targets_list):
                scale = targets_scale[idx]
                if reverse is False:
                    self.graphIO.add_in_edge(target, node, scale)
                else:
                    self.graphIO.add_out_edge(target, node, scale)
        for node in self.outs.keys():
            targets_list = self.outs[node]()
            targets_scale = self.outs[node].all_scales()
            for idx, target in enumerate(targets_list):
                scale = targets_scale[idx]
                if reverse is False:
                    self.graphIO.add_out_edge(node, target, scale)
                else:
                    self.graphIO.add_in_edge(node, target, scale)
                if need_print is True:
                    self._PrintEdge(node, target, 'out')

    def _ReplaceInputs(self, source_ops, helper, reshape_dict=None, layout='NCHW', quantized=False):
        if reshape_dict is None:
            reshape_dict = dict()
        for source_op in source_ops:
            if source_op.type in ['feed']:
                out_edges = Fluid_edger()
                for param in source_op.output_names:
                    private_data = {}
                    arg = helper.var_name_by_param(source_op, param)
                    input_node_name = self._NameNodeIn(arg)
                    for tmp_op in source_ops:
                        if tmp_op.idx != source_op.idx and arg in tmp_op.input_arg_names:
                            out_edges.add(param, self._NameNodeMid(tmp_op))
                    arg_idx = source_op.output_arg_names.index(arg)
                    shape = helper.var_shape_by_param(False, source_op, \
                        "Out", arg_idx, 'UNMODIFIED')
                    if shape[0] == -1:
                        shape[0] = 1
                    if layout == 'NCHW':
                        shape = map(int, [1] * (4 - len(shape)) + shape)
                    if input_node_name in reshape_dict.keys():
                        shape = reshape_dict[input_node_name]
                    private_data['input_shape'] = shape
                    private_data['alias'] = arg
                    self.outs[input_node_name] = out_edges
                    self._AddProtoNode(input_node_name, source_op, helper, private_data)

    def _InsertSplit(self, source_ops, helper, quantized=False):
        # If a layer has two identical output tensors, add a split layer.
        for node in self.outs.keys():
            if node.startswith('split#') is False and \
            node.startswith('increment#') is False:
                out_edges = self.outs[node]
                for param in out_edges.all_params():
                    out_targets_list = out_edges.targets(param)
                    if len(out_targets_list) > 1:
                        private_data = {}
                        private_data['split_num'] = len(out_targets_list)
                        split_node_name = 'split#' + \
                        bytes(out_edges.all_params().index(param)) + '#' + node
                        self._InitEdges(split_node_name)
                        for out_target in out_targets_list:
                            self.outs[node].rm(out_target)
                            self.ins[out_target].mv(node, split_node_name)
                            self.outs[split_node_name].add('_Out', out_target)
                        self._AddPairEdges(node, split_node_name, param, '_In')
                        self._AddProtoNode(split_node_name, None, helper, private_data, 'split_ins')

    def _Subgraph(self, starts, ends):
        """
        """
        out_idx = {}
        results = union(starts, ends)
        def outs(node):
            """
            """
            if node in self.outs.keys():
                return self.outs[node]()
            else:
                return []
        def next_out(node):
            """
            """
            next_out = ''
            if len(outs(node)) == 0:
                return -1
            elif node not in out_idx.keys():
                out_idx[node] = 0
            if out_idx[node] < len(outs(node)):
                next_out = outs(node)[out_idx[node]]
                out_idx[node] += 1
            return next_out
        for start in starts:
            cache = [start]
            while len(cache) > 0:
                target = next_out(cache[-1])
                while target != -1 and target not in results:
                    if bool(target) is True:
                        cache.append(target)
                        target = next_out(target)
                    else:
                        if cache[-1] in results:
                            results = union(results, cache)
                        break
                if target in results:
                    cache.append(target)
                    results = union(results, cache)
                cache.pop()
        return results

    def _CropGraph(self, ins_of_subgraph, outs_of_subgraph, helper, need_io=True, quantized=False):
        '''
        '''
        def all_nodes():
            '''
            '''
            all_nodes = []
            for main_node in self.ins.keys():
                all_nodes.extend(self.ins[main_node].all_targets())
            for main_node in self.outs.keys():
                all_nodes.extend(self.outs[main_node].all_targets())
            return list(set(all_nodes))
        stayed_nodes = self._Subgraph(ins_of_subgraph, outs_of_subgraph)
        all_nodes = all_nodes()
        extra_nodes = difference(all_nodes, stayed_nodes)
        for node_name in extra_nodes:
            self._RmProtoNode(node_name)
            self._ClearEdges(node_name)
            if node_name in self.graphIO.ins():
                self.graphIO.rm_in(node_name)
            if node_name in self.graphIO.outs():
                self.graphIO.rm_out(node_name)
        for node_name in ins_of_subgraph:
            if node_name in self.ins:
                self.ins[node_name].clear()
        for node_name in outs_of_subgraph:
            if node_name in self.outs:
                self.outs[node_name].clear()
        if need_io is True:
            for node_name in outs_of_subgraph:
                if node_name not in self.graphIO.outs():
                    out_node_name = node_name + '_crop_out'
                    self.ins[out_node_name] = Fluid_edger('_In', node_name)
                    self.outs[node_name] = Fluid_edger('_Out', out_node_name)
                    self.graphIO.add_out_fluid(out_node_name, node_name)
            for node_name in ins_of_subgraph:
                if node_name not in self.graphIO.ins():
                    in_node_name = node_name + '_crop_in'
                    private_data = {'input_shape': [-1, -1, -1, -1]}
                    self.ins[node_name] = Fluid_edger('_In', in_node_name)
                    self.outs[in_node_name] = Fluid_edger('_Out', node_name)
                    self._AddProtoNode(in_node_name, None, helper, private_data, 'feed')

    def _IntegrateNodes(self, main_op, main_node_name, sec_node_name, \
        helper, private_data, quantized=False):
        # Merge secondary nodes to the primary node and process the edges.
        self._RmProtoNode(main_node_name)
        self._RmProtoNode(sec_node_name)
        target_nodes_names = self.outs[sec_node_name]()
        for target_node_name in target_nodes_names:
            self.ins[target_node_name].mv(sec_node_name, main_node_name)
            self.outs[main_node_name].mv(sec_node_name, target_node_name)
            self.ins[target_node_name].rm(sec_node_name)
            self.outs[sec_node_name].rm(target_node_name)
        self.ins[sec_node_name].rm(main_node_name)
        self.outs[main_node_name].rm(sec_node_name)
        self._AddProtoNode(main_node_name, main_op, helper, private_data)

    def _DealWithBias(self, source_ops, helper, quantized=False):
        # In fluid, the bias parameter of the conv2d is split into elementwise_add.
        for source_op in source_ops:
            if source_op.type in APPEND_BIAS_OP_TYPE:
                private_data = {}
                main_node_name = self._NameNodeMid(source_op)
                if main_node_name in self.outs.keys():
                    tmp_nodes_names = self.outs[main_node_name]()
                    if len(tmp_nodes_names) == 1 and \
                    tmp_nodes_names[0].startswith('elementwise_add'):
                        elt_node_name = tmp_nodes_names[0]
                        elt_op = self._GetOp(source_ops, elt_node_name)
                        has_weights = helper.is_persistable_param(elt_op, 'Y')
                        if self._NameNodeMid(elt_op) == elt_node_name and has_weights:
                            [elt_tensor, shape] = helper.param_tensor_sh(elt_op, 'Y')
                            new_shape = [1, shape[3], 1, 1]
                            elt_tensor.set_shape(new_shape)
                            private_data['bias'] = elt_tensor
                            if main_node_name in self.scale_dict.keys():
                                private_data['scale_1'] = self.scale_dict[main_node_name]
                            self._IntegrateNodes(source_op, main_node_name, \
                                elt_node_name, helper, private_data)

    def _DealWithBatchnorm(self, source_ops, helper, quantized=False):
        # In anakin, the scale part of batchnorm layer is independent.
        for source_op in source_ops:
            if source_op.type == 'batch_norm':
                discrete_flag = True
                main_node_name = self._NameNodeMid(source_op)
                input_name = self.ins[main_node_name].target('X')
                has_scale = helper.is_persistable_param(source_op, 'Scale')
                if input_name.startswith('elementwise_add'):
                    elt_op = self._GetOp(source_ops, input_name)
                    x_of_elt = self.ins[input_name].target('X')
                    has_weights = helper.is_persistable_param(elt_op, 'Y')
                    if (x_of_elt.startswith('conv2d') or \
                        x_of_elt.startswith('depthwise_conv2d')) and has_weights:
                        discrete_flag = False
                elif input_name.startswith('conv2d') or input_name.startswith('depthwise_conv2d'):
                    discrete_flag = False
                if discrete_flag is True:
                    self._RmProtoNode(main_node_name)
                    self._AddProtoNode(main_node_name, source_op, helper, {}, 'disc_bn')
                elif has_scale is True:
                    append_node_name = main_node_name + '__scale'
                    tmp_all_targets_params = self.outs[main_node_name].targets_with_params()
                    self._InitEdges(append_node_name)
                    for [tmp_node_name, tmp_param_name] in tmp_all_targets_params:
                        self.outs[append_node_name].add(tmp_param_name, tmp_node_name)
                        self.ins[tmp_node_name].mv(main_node_name, append_node_name)
                        self.outs[main_node_name].rm(tmp_node_name)
                        self.ins[tmp_node_name].rm(main_node_name)
                    self.outs[main_node_name].add('_Scale_out', append_node_name)
                    self.ins[append_node_name].add('_Ins', main_node_name)
                    self._AddProtoNode(append_node_name, source_op, helper, {}, 'scale_of_bn')

    def _DealWithAxpy(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'elementwise_mul':
                mul_node_name = self._NameNodeMid(source_op)
                out_targets = self.outs[mul_node_name]()
                if len(out_targets) == 1 and out_targets[0].startswith('elementwise_add'):
                    add_node_name = out_targets[0]
                    self._RmProtoNode(add_node_name)
                    a_node_name = self.ins[mul_node_name].target('Y')
                    x_node_name = self.ins[mul_node_name].target('X')
                    y_node_name = self.ins[add_node_name].target('X')
                    self._ClearEdges(mul_node_name)
                    self.ins[add_node_name].clear()
                    self.outs[a_node_name].mv(mul_node_name, add_node_name)
                    self.outs[x_node_name].mv(mul_node_name, add_node_name)
                    self.ins[add_node_name].add('A', a_node_name)
                    self.ins[add_node_name].add('X', x_node_name)
                    self.ins[add_node_name].add('Y', y_node_name)
                    self._RmProtoNode(mul_node_name)
                    self._AddProtoNode(add_node_name, None, helper, {}, 'axpy')

    def _DealWithPriorBox(self, source_ops, helper, is_dev_v2=True, quantized=False):
        nodes_to_del = []
        for source_op in source_ops:
            if source_op.type == 'prior_box':
                if is_dev_v2 is True:
                    axis = 2
                else:
                    axis = 3
                private_data = {"axis": axis}
                pb_node_name = self._NameNodeMid(source_op)
                br_node_name = self.outs[pb_node_name].target('Boxes')
                vr_node_name = self.outs[pb_node_name].target('Variances')
                bc_node_name = self.outs[br_node_name].target('Out')
                vc_node_name = self.outs[vr_node_name].target('Out')
                boxcoder_node_name = self.outs[bc_node_name].target('Out')
                self.outs[pb_node_name].mv(br_node_name, bc_node_name)
                self.outs[pb_node_name].rm(vr_node_name)
                self.ins[bc_node_name].mv(br_node_name, pb_node_name)
                self.ins[boxcoder_node_name].rm(vc_node_name)
                for node_name in [br_node_name, vr_node_name, vc_node_name]:
                    if node_name not in nodes_to_del:
                        nodes_to_del.append(node_name)
                input_node_name = self.ins[pb_node_name].target('Input')
                image_node_name = self.ins[pb_node_name].target('Image')
                self.ins[pb_node_name].rm(input_node_name)
                self.ins[pb_node_name].rm(image_node_name)
                self.ins[pb_node_name].add('Input', input_node_name)
                self.ins[pb_node_name].add('Image', image_node_name)
                self._RmProtoNode(bc_node_name)
                self._AddProtoNode(bc_node_name, None, helper, private_data, \
                    'concat_btw_priorbox_boxcoder')
        for node_name in nodes_to_del:
            self._RmProtoNode(node_name)
            self._ClearEdges(node_name)

    def _DealWithDetectionOutput(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'box_coder':
                bc_node_name = self._NameNodeMid(source_op)
                out_targets = self.outs[bc_node_name]()
                if len(out_targets) == 1 and out_targets[0].startswith('multiclass_nms'):
                    private_data = {}
                    private_data['code_type'] = helper.attr_data(source_op, 'code_type')
                    bc_out_arg = helper.var_name_by_param(source_op, 'OutputBox')
                    for tmp_op in source_ops:
                        if tmp_op.idx != source_op.idx and bc_out_arg in tmp_op.input_arg_names:
                            nms_op = tmp_op
                    nms_node_name = out_targets[0]
                    loc_node_name = self.ins[bc_node_name].target('TargetBox')
                    conf_node_name = self.ins[nms_node_name].target('Scores')
                    box_node_name = self.ins[bc_node_name].target('PriorBox')
                    self._ClearEdges(bc_node_name)
                    self.ins[nms_node_name].clear()
                    self.outs[loc_node_name].mv(bc_node_name, nms_node_name)
                    self.outs[box_node_name].mv(bc_node_name, nms_node_name)
                    self.ins[nms_node_name].add('mbox_loc', loc_node_name)
                    self.ins[nms_node_name].add('mbox_conf_flatten', conf_node_name)
                    self.ins[nms_node_name].add('mbox_prior_box', box_node_name)
                    self._RmProtoNode(bc_node_name)
                    self._RmProtoNode(nms_node_name)
                    self._AddProtoNode(nms_node_name, nms_op, helper, \
                        private_data, 'multiclass_nms')

    def _DealWithMultiFC(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'sum':
                sum_node_name = self._NameNodeMid(source_op)
                mul_names_list = self.ins[sum_node_name].targets('X')
                elt_node_name = self.outs[sum_node_name].target('Out')
                if elt_node_name.startswith('elementwise_add') and len(mul_names_list) > 1:
                    elt_op = self._GetOp(source_ops, elt_node_name)
                    elt_has_weights = helper.is_persistable_param(elt_op, 'Y')
                    fc_flag = True
                    for mul_node_name in mul_names_list:
                        if mul_node_name.startswith('mul') is False:
                            fc_flags = False
                    if fc_flag and elt_has_weights:
                        private_data = {}
                        first_mul_name = mul_names_list[0]
                        first_mul_op = self._GetOp(source_ops, first_mul_name)
                        in_of_mul_name = self.ins[first_mul_name].target('X')
                        out_of_elt_name = self.outs[elt_node_name].target('Out')
                        self.outs[sum_node_name].mv(elt_node_name, out_of_elt_name)
                        self.ins[out_of_elt_name].mv(elt_node_name, sum_node_name)
                        self._ClearEdges(elt_node_name)
                        [elt_tensor, shape] = helper.param_tensor_sh(elt_op, 'Y')
                        new_shape = [1, shape[3], 1, 1]
                        elt_tensor.set_shape(new_shape)
                        private_data['bias'] = elt_tensor
                        self._RmProtoNode(elt_node_name)
                        self._RmProtoNode(first_mul_name)
                        self._AddProtoNode(first_mul_name, first_mul_op, helper, private_data)

    def _DealWithGru(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'gru':
                private_data = {}
                gru_flags = [False, False]
                gru_node_name = self._NameNodeMid(source_op)
                mul_node_name = str()
                gru_op = self._GetOp(source_ops, gru_node_name)
                input_list_of_gru = self.ins[gru_node_name].targets('Input')
                if len(input_list_of_gru) == 1 and input_list_of_gru[0]. \
                startswith('elementwise_add'):
                    elt_node_name = input_list_of_gru[0]
                    elt_op = self._GetOp(source_ops, elt_node_name)
                    has_weights = helper.is_persistable_param(elt_op, 'Y')
                    if has_weights is True:
                        private_data['np_bias_x'] = helper.np_param(elt_op, 'Y')
                        gru_flags[0] = True
                    input_list_of_elt = self.ins[elt_node_name].targets('X')
                    if len(input_list_of_elt) == 1 and input_list_of_elt[0].startswith('mul'):
                        mul_node_name = input_list_of_elt[0]
                elif len(input_list_of_gru) == 1 and input_list_of_gru[0].startswith('mul'):
                    mul_node_name = input_list_of_gru[0]
                    private_data['np_bias_x'] = 0
                if bool(mul_node_name):
                    mul_op = self._GetOp(source_ops, mul_node_name)
                    if helper.var_name_by_param(mul_op, 'Y').startswith('fc'):
                        if helper.attr_data(mul_op, 'x_num_col_dims') == 1:
                            input_list_of_mul = self.ins[mul_node_name].targets('X')
                            input_name_of_mul = input_list_of_mul[0]
                            private_data['np_weight_x'] = helper.np_param(mul_op, 'Y')
                            gru_flags[1] = True
                        else:
                            raise NameError('ERROR: Axis of GRU_FC must be 1.')
                if gru_flags[1]:
                    self.outs[input_name_of_mul].mv(mul_node_name, gru_node_name)
                    self._AddProtoNode(gru_node_name, gru_op, helper, private_data)
                    if gru_flags[0]:
                        self.ins[gru_node_name].mv(elt_node_name, input_name_of_mul)
                        nodes_to_del = [mul_node_name, elt_node_name, gru_node_name]
                    else:
                        self.ins[gru_node_name].mv(mul_node_name, input_name_of_mul)
                        nodes_to_del = [mul_node_name, gru_node_name]
                    for node_to_del_name in nodes_to_del:
                        self._RmProtoNode(node_to_del_name)
                        if node_to_del_name is not gru_node_name:
                            self._ClearEdges(node_to_del_name)

    def _SearchBilstm(self, source_ops, helper, quantized=False):
        comp = Fluid_comparator(helper)
        lstm_ops = []
        for source_op in source_ops:
            if source_op.type == 'lstm':
                lstm_ops.append(source_op)
        if len(lstm_ops) == 2:
            lstm_a = lstm_ops[0]
            lstm_b = lstm_ops[1]
            same_bias = comp.compare_by_param(lstm_a, lstm_b, 'Bias')
            same_weight = comp.compare_by_param(lstm_a, lstm_b, 'Weight')
            if same_bias and same_weight:
                return True
            else:
                return False
        else:
            return False

    def _DealWithLstm(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'lstm':
                private_data = {}
                lstm_flags = [False, False]
                lstm_node_name = self._NameNodeMid(source_op)
                lstm_op = self._GetOp(source_ops, lstm_node_name)
                input_list_of_lstm = self.ins[lstm_node_name].targets('Input')
                input_list = []
                if len(input_list_of_lstm) == 1:
                    in_lstm_node_name = input_list_of_lstm[0]
                    if input_list_of_lstm[0].split('#')[0] == 'elementwise_add':
                        elt_op = self._GetOp(source_ops, in_lstm_node_name)
                        has_weights = helper.is_persistable_param(elt_op, 'Y')
                        if has_weights is True:
                            private_data['np_flat_fc_bias'] = helper.np_param(elt_op, 'Y')
                            lstm_flags[0] = True
                        input_list = self.ins[in_lstm_node_name].targets('X')
                    elif input_list_of_lstm[0].split('#')[0] == 'mul':
                        private_data['np_flat_fc_bias'] = None
                        input_list = input_list_of_lstm
                        lstm_flags[0] = True
                if lstm_flags[0] is True and len(input_list) == 1:
                    if input_list[0].split('#')[0] == 'mul':
                        mul_node_name = input_list[0]
                        mul_op = self._GetOp(source_ops, mul_node_name)
                        #if helper.var_name_by_param(mul_op, 'Y').startswith('fc'):
                        if helper.attr_data(mul_op, 'x_num_col_dims') == 1:
                            input_list_of_mul = self.ins[mul_node_name].targets('X')
                            input_name_of_mul = input_list_of_mul[0]
                            [w_np, w_sh] = helper.data_with_shape_by_param(mul_op, 'Y', \
                                    False, None, 0, False)
                            private_data['np_flat_fc_weight'] = w_np
                            private_data['np_fc_outdim'] = w_sh[3]
                            lstm_flags[1] = True
                        else:
                            raise NameError('ERROR: Axis of LSTM_FC must be 1.')
                if lstm_flags == [True, True]:
                    self.outs[input_name_of_mul].mv(mul_node_name, lstm_node_name)
                    self.ins[lstm_node_name].mv(in_lstm_node_name, input_name_of_mul)
                    if in_lstm_node_name == mul_node_name:
                        nodes_to_del = [mul_node_name, lstm_node_name]
                    else:
                        nodes_to_del = [mul_node_name, in_lstm_node_name, lstm_node_name]
                    for node_to_del_name in nodes_to_del:
                        self._RmProtoNode(node_to_del_name)
                        if node_to_del_name is not lstm_node_name:
                            self._ClearEdges(node_to_del_name)
                    self._AddProtoNode(lstm_node_name, lstm_op, helper, private_data)

    def _DealWithCast(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'cast':
                if helper.attr_data(source_op, 'out_dtype') == 5:
                    cast_node_name = self._NameNodeMid(source_op)
                    if cast_node_name in self.ins:
                        input_name_of_cast = self.ins[cast_node_name].target('X')
                        if input_name_of_cast.startswith('top_k') is False:
                            output_name_of_cast = self.outs[cast_node_name].target('Out')
                            self.outs[input_name_of_cast].mv(cast_node_name, output_name_of_cast)
                            self.ins[output_name_of_cast].mv(cast_node_name, input_name_of_cast)
                            self._RmProtoNode(cast_node_name)
                            self._ClearEdges(cast_node_name)
                    else:
                        print 'Cannot find the layer corresponding to cast.'
                else:
                    raise NameError('The out type of cast must be float32.')

    def _DealWithArgmax(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'top_k':
                private_data = {}
                topk_node_name = self._NameNodeMid(source_op)
                out_list = self.outs[topk_node_name].targets('Out')
                index_list = self.outs[topk_node_name].targets('Indices')
                if len(index_list) > 0:
                    if len(out_list) == 1 and index_list[0].startswith('cast'):
                        private_data['out_max_val'] = True
                        idxcast_node_name = index_list[0]
                        output_name_of_idxcast = self.outs[idxcast_node_name].target('Out')
                        if output_name_of_idxcast == out_list[0] and \
                        out_list[0].startswith('concat'):
                            concat_node_name = out_list[0]
                            output_name_of_concat = self.outs[concat_node_name].target('Out')
                            self.outs[topk_node_name].rm(idxcast_node_name)
                            self.outs[topk_node_name].mv(concat_node_name, output_name_of_concat)
                            self.ins[output_name_of_concat].mv(concat_node_name, topk_node_name)
                            for node_to_del_name in [concat_node_name, idxcast_node_name]:
                                self._RmProtoNode(node_to_del_name)
                                self._ClearEdges(node_to_del_name)
                        elif output_name_of_idxcast != out_list[0]:
                            if output_name_of_idxcast.endswith('_gout') and \
                            out_list[0].endswith('_gout'):
                                gout_node_name = out_list[0]
                                idx_gout_node_name = output_name_of_idxcast
                                self.outs[topk_node_name].rm(idxcast_node_name)
                                for node_to_del_name in [idx_gout_node_name, idxcast_node_name]:
                                    self._RmProtoNode(node_to_del_name)
                                    self._ClearEdges(node_to_del_name)
                                self.graphIO.rm_out(idx_gout_node_name)
                    elif len(out_list) == 0:
                        private_data['out_max_val'] = False
                        self._DealWithCast(source_ops, helper)
                    else:
                        raise NameError('ERROR: Unknown top_k layer.')
                    self._RmProtoNode(topk_node_name)
                    self._AddProtoNode(topk_node_name, source_op, helper, private_data)

    def _RefreshReshape(self, source_ops, helper, need_assign=False, quantized=False):
        for source_op in source_ops:
            if source_op.type in ['reshape', 'reshape2']:
                reshape_node_name = self._NameNodeMid(source_op)
                # Make sure this node exists in this graph.
                if reshape_node_name in self.ins:
                    shape_inputs = self.ins[reshape_node_name].targets('Shape')
                    tensor_inputs = self.ins[reshape_node_name].targets('X')
                    if len(shape_inputs) == 1 and len(tensor_inputs) == 1:
                        self.ins[reshape_node_name].rm(shape_inputs[0])
                        if shape_inputs[0].split('#')[0] != 'assign_value' \
                        or need_assign is True:
                            self.ins[reshape_node_name].add('Shape', shape_inputs[0])
                        else:
                            self._RmProtoNode(shape_inputs[0])
                            self._ClearEdges(shape_inputs[0])

    def _CutReshape(self, reshape_node_name, quantized=False):
        branch = []
        branch.append(reshape_node_name)
        shape_inputs = self.ins[reshape_node_name].targets('Shape')
        tensor_input = self.ins[reshape_node_name].target('X')
        tensor_output = self.outs[reshape_node_name].target('Out')
        if len(shape_inputs) == 1:
            branch.append(shape_inputs[0])
        if len(branch) == 2 and branch[1].split('#')[0] == 'split':
            split_node_name = branch[1]
            self.outs[split_node_name].rm(reshape_node_name)
            self.ins[reshape_node_name].rm(split_node_name)
            if len(self.outs[split_node_name].targets('_Out')) == 0:
                input_of_split = self.ins[split_node_name].target('_In')
                branch.append(input_of_split)
                self._RmProtoNode(split_node_name)
                self._ClearEdges(split_node_name)
        elif len(branch) == 2 and branch[1].split('#')[0] == 'shape':
            shape_node_name = branch[1]
            input_of_shape = self.ins[shape_node_name].targets('Input')
            assert len(input_of_shape) == 1
            self.outs[input_of_shape[0]].rm(shape_node_name)
            self.ins[reshape_node_name].rm(shape_node_name)
            self._RmProtoNode(shape_node_name)
            self._ClearEdges(shape_node_name)
        elif len(branch) == 2 and branch[1].split('#')[0] == 'assign_value':
            assign_node_name = branch[1]
            self.ins[reshape_node_name].rm(assign_node_name)
            self._RmProtoNode(assign_node_name)
            self._ClearEdges(assign_node_name)
        elif len(branch) == 2 and branch[1].startswith('input'):
            raise NameError('ERROR: None-split input of Softmax has not supported.')
        else:
            pass
        self.outs[tensor_input].mv(reshape_node_name, tensor_output)
        self.ins[tensor_output].mv(reshape_node_name, tensor_input)
        self._RmProtoNode(reshape_node_name)
        self._ClearEdges(reshape_node_name)
        if len(branch) == 3 and branch[2].startswith('input'):
            input_node_name = branch[2]
            self._RmProtoNode(input_node_name)
            self._ClearEdges(input_node_name)

    def _RefreshSplit(self, split_node_name, helper, quantized=False):
        outputs_of_split = self.outs[split_node_name].targets('_Out')
        inputs_of_split = self.ins[split_node_name].targets('_In')
        assert len(inputs_of_split) < 2
        split_num = len(outputs_of_split)
        if split_num == 0:
            print 'WARNING: RefeshSplit num is equal to zero.'
        elif split_num == 1:
            self.ins[outputs_of_split[0]].mv(split_node_name, inputs_of_split[0])
            self.outs[inputs_of_split[0]].mv(split_node_name, outputs_of_split[0])
            self._RmProtoNode(split_node_name)
            self._ClearEdges(split_node_name)
        else:
            private_data = {'split_num': split_num}
            self._RmProtoNode(split_node_name)
            self._AddProtoNode(split_node_name, None, helper, private_data, 'split_ins')

    def _DealWithSoftmax(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'softmax':
                softmax_node_name = self._NameNodeMid(source_op)
                outs_of_softmax = self.outs[softmax_node_name].targets('Out')
                ins_of_softmax = self.ins[softmax_node_name].targets('X')
                if outs_of_softmax[0].split('#')[0] in ['reshape', 'reshape2']:
                    if ins_of_softmax[0].split('#')[0] in ['reshape', 'reshape2'] or \
                    ins_of_softmax[0].split('#')[0] in ['flatten', 'flatten2']:
                        private_data = {}
                        private_data['axis'] = 3
                        self._CutReshape(outs_of_softmax[0])
                        self._CutReshape(ins_of_softmax[0])
                        self._RmProtoNode(softmax_node_name)
                        self._AddProtoNode(softmax_node_name, source_op, helper, private_data)
                        ins_of_softmax = self.ins[softmax_node_name].targets('X')
                        assert len(ins_of_softmax) == 1
                        if ins_of_softmax[0].startswith('split'):
                            self._RefreshSplit(ins_of_softmax[0], helper)

    def _DealWithMatmal(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'matmul':
                matmul_node_name = self._NameNodeMid(source_op)
                x_input_name = self.ins[matmul_node_name].target('X')
                y_input_name = self.ins[matmul_node_name].target('Y')
                flag = False
                coeff = 1.0
                for node_name in [x_input_name, y_input_name]:
                    if node_name.startswith('scale') or node_name.startswith('dropout'):
                        op = self._GetOp(source_ops, node_name)
                        if node_name.startswith('scale'):
                            scale = helper.attr_data(op, 'scale')
                        elif node_name.startswith('dropout'):
                            scale = 1 - helper.attr_data(op, 'dropout_prob')
                        input_node = self.ins[node_name].target('X')
                        self.outs[input_node].mv(node_name, matmul_node_name)
                        self.ins[matmul_node_name].mv(node_name, input_node)
                        self._RmProtoNode(node_name)
                        self._ClearEdges(node_name)
                        coeff = coeff * scale
                        flag = True
                if flag is True:
                    private_data = {}
                    private_data['coeff'] = coeff
                    self._RmProtoNode(matmul_node_name)
                    self._AddProtoNode(matmul_node_name, source_op, helper, private_data)

    def _DealWithDiscBatchNorm(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'batch_norm':
                discrete_flag = True
                bn_node_name = self._NameNodeMid(source_op)
                input_name = self.ins[bn_node_name].target('X')
                if input_name.startswith('elementwise_add'):
                    input_of_elt = self.ins[input_name].target('X')
                    if input_of_elt.startswith('conv2d'):
                        discrete_flag = False
                elif input_name.startswith('conv2d'):
                    discrete_flag = False
                if discrete_flag is True:
                    self._RmProtoNode(bn_node_name)
                    self._AddProtoNode(bn_node_name, source_op, helper, {}, 'disc_bn')

    def _DealWithSSD(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'softmax':
                private_data = dict()
                sm_node_name = self._NameNodeMid(source_op)
                outs_of_sm = self.outs[sm_node_name].targets('Out')
                if outs_of_sm[0].startswith('transpose'):
                    ts_node_name = outs_of_sm[0]
                    out_of_ts = self.outs[ts_node_name].target('Out')
                    self.outs[sm_node_name].mv(ts_node_name, out_of_ts)
                    self.ins[out_of_ts].mv(ts_node_name, sm_node_name)
                    self._RmProtoNode(ts_node_name)
                    self._ClearEdges(ts_node_name)
                private_data['axis'] = 2
                self._RmProtoNode(sm_node_name)
                self._AddProtoNode(sm_node_name, source_op, helper, private_data, 'softmax')


    def _DealWithPixelShuffle(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type in ['transpose', 'transpose2']:
                axis = helper.attr_data(source_op, 'axis')
                if axis == [0, 1, 4, 2, 5, 3]:
                    private_data = dict()
                    ts_node_name = self._NameNodeMid(source_op)
                    in_of_transpose = self.ins[ts_node_name].target('X')
                    out_of_transpose = self.outs[ts_node_name].target('Out')
                    if in_of_transpose.startswith('reshape') and \
                    out_of_transpose.startswith('reshape'):
                        in_reshape_op = self._GetOp(source_ops, in_of_transpose)
                        out_reshape_op = self._GetOp(source_ops, out_of_transpose)
                        in_shape = helper.attr_data(in_reshape_op, 'shape')
                        out_shape = helper.attr_data(out_reshape_op, 'shape')
                        private_data['factor'] = out_shape[-1] / in_shape[-1]
                        in_first_reshape = self.ins[in_of_transpose].target('X')
                        out_last_reshape = self.outs[out_of_transpose].target('Out')
                        self.outs[in_first_reshape].mv(in_of_transpose, ts_node_name)
                        self.outs[ts_node_name].mv(out_of_transpose, out_last_reshape)
                        self.ins[out_last_reshape].mv(out_of_transpose, ts_node_name)
                        self.ins[ts_node_name].mv(in_of_transpose, in_first_reshape)
                        self._RmProtoNode(in_of_transpose)
                        self._RmProtoNode(out_of_transpose)
                        self._ClearEdges(in_of_transpose)
                        self._ClearEdges(out_of_transpose)
                        self._RmProtoNode(ts_node_name)
                        self._AddProtoNode(ts_node_name, None, helper, \
                            private_data, 'pixel_shuffle')

    def _DealWithShuffleChannel(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type in ['transpose', 'transpose2']:
                axis = helper.attr_data(source_op, 'axis')
                if axis == [0, 2, 1, 3, 4]:
                    private_data = dict()
                    ts_node_name = self._NameNodeMid(source_op)
                    in_of_transpose = self.ins[ts_node_name].target('X')
                    out_of_transpose = self.outs[ts_node_name].target('Out')
                    if in_of_transpose.startswith('reshape') and \
                    out_of_transpose.startswith('reshape'):
                        in_reshape_op = self._GetOp(source_ops, in_of_transpose)
                        out_reshape_op = self._GetOp(source_ops, out_of_transpose)
                        in_shape = helper.attr_data(in_reshape_op, 'shape')
                        out_shape = helper.attr_data(out_reshape_op, 'shape')
                        private_data['group'] = out_shape[-3] / in_shape[-3]
                        in_first_reshape = self.ins[in_of_transpose].target('X')
                        out_last_reshape = self.outs[out_of_transpose].target('Out')
                        self.outs[in_first_reshape].mv(in_of_transpose, ts_node_name)
                        self.outs[ts_node_name].mv(out_of_transpose, out_last_reshape)
                        self.ins[out_last_reshape].mv(out_of_transpose, ts_node_name)
                        self.ins[ts_node_name].mv(in_of_transpose, in_first_reshape)
                        self._RmProtoNode(in_of_transpose)
                        self._RmProtoNode(out_of_transpose)
                        self._ClearEdges(in_of_transpose)
                        self._ClearEdges(out_of_transpose)
                        self._RmProtoNode(ts_node_name)
                        self._AddProtoNode(ts_node_name, None, helper, \
                            private_data, 'shuffle_channel')

    def _DealWithAnchorGenerator(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'anchor_generator':
                private_data = dict()
                ag_node_name = self._NameNodeMid(source_op)
                out_edges = self.outs[ag_node_name]
                for param in out_edges.all_params():
                    arg = helper.args_by_output_param(source_op, param)
                    out_target = out_edges.target(param)
                    if out_target.startswith('generate_proposals') is False:
                        raise NameError('ERROR: Unknown output of AnchorGenerator.')
                    private_data['split_num'] = 1
                    split_node_name = 'split#' + \
                    bytes(out_edges.all_params().index(param)) + '#' + ag_node_name
                    self._InitEdges(split_node_name)
                    self.outs[ag_node_name].reset_target_by_param(param, split_node_name)
                    in_edges = self.ins[out_target]
                    in_op = self._GetOp(source_ops, out_target)
                    for in_param in in_edges.all_params():
                        in_arg = helper.args_by_input_param(in_op, in_param)
                        if in_arg == arg:
                            self.ins[out_target].reset_target_by_param(in_param, split_node_name)
                    self.outs[split_node_name].add('_Out', out_target)
                    self._AddPairEdges(ag_node_name, split_node_name, param, '_In')
                    self._AddProtoNode(split_node_name, None, helper, private_data, 'split_ins')

    def _DealWithGenerateProposals(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'generate_proposals':
                gp_node_name = self._NameNodeMid(source_op)
                targets = self.outs[gp_node_name].all_targets()
                if len(targets) == 1 is True or targets[0].startswith('split#') is True:
                    arg_node_name = 'temp_out_of_generate_proposals'
                    self.graph_outs.append(arg_node_name)
                    self.graphIO.add_out_fluid(arg_node_name, \
                        gp_node_name)
                    self.outs[gp_node_name].add('temp_out', arg_node_name)
                    self.ins[arg_node_name] = Fluid_edger(bytes(source_op.idx), \
                        gp_node_name)
                    ''' 
                    anchors_in = self.ins[gp_node_name].target('Anchors')
                    bboxdeltas_in = self.ins[gp_node_name].target('BboxDeltas')
                    iminfo_in = self.ins[gp_node_name].target('ImInfo')
                    scores_in = self.ins[gp_node_name].target('Scores')
                    variances_in = self.ins[gp_node_name].target('Variances')
                    targets_in = [anchors_in, bboxdeltas_in, iminfo_in, \
                    scores_in, variances_in]
                    for target_in in targets_in:
                        self.ins[gp_node_name].rm(target_in)
                    self.ins[gp_node_name].add('Anchors', anchors_in)
                    self.ins[gp_node_name].add('BboxDeltas', bboxdeltas_in)
                    self.ins[gp_node_name].add('ImInfo', iminfo_in)
                    self.ins[gp_node_name].add('Scores', scores_in)
                    self.ins[gp_node_name].add('Variances', variances_in)
                    '''

    def _DelIncInQuantize(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type in ['increment']:
                inc_node_name = self._NameNodeMid(source_op)
                self._RmProtoNode(inc_node_name)
                self._ClearEdges(inc_node_name)

    def _DealWithQuantize(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type in FLUID_QUANTIZE_LAYERS:
                qt_node_name = self._NameNodeMid(source_op)
                in_of_qt = self.ins[qt_node_name].target('X')
                out_param_of_in = self.outs[in_of_qt].all_params()[0]
                outs_of_qt = self.outs[qt_node_name].targets('Out')
                qt_node = self._GetOp(source_ops, qt_node_name)
                in_scale = helper.data_with_shape_by_param(qt_node, 'InScale')[0][0]
                in_scale = in_scale / 127
                self.outs[in_of_qt].rm(qt_node_name)
                for out_of_qt in outs_of_qt:
                    op_out_q = self._GetOp(source_ops, out_of_qt)
                    param_name = out_param_of_in
                    self.outs[in_of_qt].add(param_name, out_of_qt, None, in_scale)
                    self.ins[out_of_qt].mv(qt_node_name, in_of_qt)
                    self.ins[out_of_qt].set_scale(in_of_qt, in_scale)
                self._RmProtoNode(qt_node_name)
                self._ClearEdges(qt_node_name)
        self._DelIncInQuantize(source_ops, helper, quantized)

    def _DealWithDequantize(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type in FLUID_DEQUANTIZE_LAYERS:
                private_data = dict()
                qt_node_name = self._NameNodeMid(source_op)
                qt_node = self._GetOp(source_ops, qt_node_name)
                in_of_qt = self.ins[qt_node_name].target('X')
                out_of_qt = self.outs[qt_node_name].target('Out')
                op_in_q = self._GetOp(source_ops, in_of_qt)
                scale_of_weight = helper.attr_data(source_op, 'max_range')
                scale_of_weight = 127 / scale_of_weight
                self.scale_dict[in_of_qt] = [scale_of_weight]
                private_data['scale_1'] = self.scale_dict[in_of_qt]
                scale = helper.data_with_shape_by_param(qt_node, 'Scale')[0][0]
                scale = scale / 127
                self.outs[in_of_qt].mv(qt_node_name, out_of_qt)
                self.outs[in_of_qt].set_scale(out_of_qt, scale)
                self.ins[out_of_qt].mv(qt_node_name, in_of_qt)
                self.ins[out_of_qt].set_scale(in_of_qt, scale)
                self._RmProtoNode(qt_node_name)
                self._ClearEdges(qt_node_name)
                self._RmProtoNode(in_of_qt)
                self._AddProtoNode(in_of_qt, op_in_q, helper, private_data)

    def _DealWithRoiAlign(self, source_ops, helper, quantized=False):
        for source_op in source_ops:
            if source_op.type == 'roi_align':
                ra_node_name = self._NameNodeMid(source_op)
                x_in_of_ra = self.ins[ra_node_name].target('X')
                rois_in_of_ra = self.ins[ra_node_name].target('ROIs')
                self.ins[ra_node_name].rm(x_in_of_ra)
                self.ins[ra_node_name].rm(rois_in_of_ra)
                self.ins[ra_node_name].add('X', x_in_of_ra, None)
                self.ins[ra_node_name].add('ROIs', rois_in_of_ra, None)

    def _NewCommonLayer(self,
                        source_ops,
                        in_target,
                        in_param,
                        out_target,
                        out_param,
                        layer_type,
                        private_data,
                        helper,
                        insert_mode=True,
                        quantized=False):
        main_layer = layer_type + '_after_' + in_target
        if insert_mode is True:
            if in_target in self.ins[out_target].all_targets() and \
            out_target in self.outs[in_target].all_targets():
                self.ins[out_target].mv(in_target, main_layer)
                self.outs[in_target].mv(out_target, main_layer)
            else:
                raise NameError('ERROR: Usage of InsertCommonLayer has not supported.')
        else:
            self.ins[out_target].add(in_param + '_insert', main_layer)
            self.outs[in_target].add(out_param + '_insert', main_layer)
        self.ins[main_layer] = Fluid_edger(in_param, in_target)
        self.outs[main_layer] = Fluid_edger(out_param, out_target)
        self._AddProtoNode(main_layer, None, helper, private_data, layer_type)

    def _ParseNetwork(self, source_ops, helper, quantized=False):
        self._ParseBase(source_ops, helper)
        if self.NetType == "FLUIDBASE":
            pass
        else:
            reshape_dict = {}
            if self.NetType == "OCR":
                reshape_dict['input_0'] = [1, 1, 48, 1500]
            elif self.NetType == "ROUTEDNN":
                reshape_dict['input_0'] = [1, 37, 1, 1]
            self._ReplaceInputs(source_ops, helper, reshape_dict)
            self._DealWithQuantize(source_ops, helper)
            self._DealWithDequantize(source_ops, helper)
            self._InsertSplit(source_ops, helper)
            self._DealWithBias(source_ops, helper)
            self._DealWithGru(source_ops, helper)
            self._DealWithLstm(source_ops, helper)
            self._DealWithBatchnorm(source_ops, helper)
            self._DealWithMultiFC(source_ops, helper)
            self._DealWithArgmax(source_ops, helper)
            self._DealWithAxpy(source_ops, helper)
            self._DealWithPixelShuffle(source_ops, helper)
            self._DealWithShuffleChannel(source_ops, helper)
            if self.NetType == "FASTRCNN":
                self._DealWithAnchorGenerator(source_ops, helper)
                self._DealWithGenerateProposals(source_ops, helper)
                self._DealWithRoiAlign(source_ops, helper)
            if self.NetType == "SSD":
                self._DealWithPriorBox(source_ops, helper)
                self._DealWithDetectionOutput(source_ops, helper)
                self._DealWithSoftmax(source_ops, helper)
                self._DealWithSSD(source_ops, helper)
                self._RefreshReshape(source_ops, helper)
        if self.Debug == 'IN':
            self._Graph(True, False)
        else:
            self._Graph(False, False)

    def _Parsing(self):
        with fluid.scope_guard(self.scope):
            model_abs_path = os.path.join(self.ModelPath, 'model')
            param_abs_path = os.path.join(self.ModelPath, 'params')
            if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
                [self.net_program, feed_target_names, fetch_targets] = \
                fluid.io.load_inference_model(self.ModelPath, self.exe, 'model', 'params')
            else:
                [self.net_program, feed_target_names, fetch_targets] = \
                fluid.io.load_inference_model(self.ModelPath, self.exe)

            global_block = self.net_program.global_block()
            source_ops = list(global_block.ops)
            helper = Fluid_helper(self.scope, global_block)

            self._ParseNetwork(source_ops, helper)

            return self.graphIO
