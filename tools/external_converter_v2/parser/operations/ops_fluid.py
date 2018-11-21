#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
from op import OpsParam, OpsRegister
from op_io import *


OpsRegister.Register("elementwise_mul").set_attr()
OpsRegister.Register("depthwise_conv2d").set_attr()
OpsRegister.Register("transpose").set_attr()
OpsRegister.Register("reshape").set_attr()
OpsRegister.Register("concat").set_attr()
OpsRegister.Register("box_coder").set_attr()

OpsRegister.Register("im2sequence").set_attr()
OpsRegister.Register("sum").set_attr()
OpsRegister.Register("top_k").set_attr()
OpsRegister.Register("ctc_align").set_attr()
OpsRegister.Register("cast").set_attr()
OpsRegister.Register("elementwise_add_fulid").set_attr()

OpsRegister.Register("lookup_table").set_attr()
OpsRegister.Register("lstm").set_attr()
OpsRegister.Register("sequence_pool").set_attr()
OpsRegister.Register("tanh").set_attr()

OpsRegister.Register("sequence_conv").set_attr()
OpsRegister.Register("stanh").set_attr()

OpsRegister.Register("matmul").set_attr()
OpsRegister.Register("layer_norm").set_attr()
OpsRegister.Register("dropout").set_attr()
OpsRegister.Register("scale").set_attr()
OpsRegister.Register("norm").set_attr()

OpsRegister.Register("lod_reset").set_attr()
OpsRegister.Register("fill_constant").set_attr()
OpsRegister.Register("lod_rank_table").set_attr()
OpsRegister.Register("max_sequence_len").set_attr()
OpsRegister.Register("less_than").set_attr()
OpsRegister.Register("lod_tensor_to_array").set_attr()
OpsRegister.Register("write_to_array").set_attr()
OpsRegister.Register("reorder_lod_tensor_by_rank").set_attr()
OpsRegister.Register("while").set_attr()
OpsRegister.Register("array_to_lod_tensor").set_attr()

OpsRegister.Register("assign_value").set_attr()
OpsRegister.Register("shape").set_attr()

OpsRegister.Register("fake_quantize_abs_max").set_attr()
OpsRegister.Register("fake_dequantize_max_abs").set_attr()

