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

OpsRegister.Register("assign").set_attr()
OpsRegister.Register("assign_value").set_attr()
OpsRegister.Register("shape").set_attr()

OpsRegister.Register("fake_quantize_abs_max").set_attr()
OpsRegister.Register("fake_dequantize_max_abs").set_attr()
OpsRegister.Register("fake_quantize_range_abs_max").set_attr()
OpsRegister.Register("fake_dequantize_range_max_abs").set_attr()

OpsRegister.Register("increment").set_attr()

OpsRegister.Register("fusion_dropout_add_ln_quant").set_attr()
OpsRegister.Register("dequantize_max_abs_rowwise").set_attr()
OpsRegister.Register("quantize_abs_max_rowwise").set_attr()
OpsRegister.Register("fusion_add_relu_dropout_quant").set_attr()
OpsRegister.Register("fill_constant_batch_size_like").set_attr()
OpsRegister.Register("beam_search_decode").set_attr()

OpsRegister.Register('reduce').set_attr(
    reduce_type=str(),
    keep_dim=bool(),
    reduce_dim=list(),
    reduce_all=bool(),
    coeff=float(),
)
OpsRegister.Register('arg_max').set_attr(
    out_max_val=bool(),
    top_k=int(),
    axis=int(),
)
OpsRegister.Register('sequence_expand').set_attr(
    ref_level=int(),
)
OpsRegister.Register('eltwise').set_attr(
    type=str(),
    coeff=float(),
)
OpsRegister.Register('cast').set_attr(
    int_type=int(),
    out_type=int(),
)
OpsRegister.Register('yolo_box').set_attr(
    anchors=list(),
    class_num=int(),
    conf_thresh=float(),
    downsample_ratio=int(),
)
OpsRegister.Register('slice').set_attr(
    slice_dim=int(),
    slice_point=list(),
    axis=int(),
)
OpsRegister.Register('box_coder').set_attr(
    axis=int(),
    box_normalized=bool(),
    variance=list(),
)
OpsRegister.Register('GroupNormal').set_attr(
    has_scale=bool(),
    has_bias=bool(),
    eps=float(),
    group=int(),
)
OpsRegister.Register('slice_v2').set_attr(
    starts=list(),
    ends=list(),
    axes=list(),
)
OpsRegister.Register('arithmetic').set_attr(
    op_type=int(),
)
OpsRegister.Register('aligned_mat_mul').set_attr(
    is_transpose_X=bool(),
    is_transpose_Y=bool(),
    scale=float(),
)
OpsRegister.Register('attention_padding_mask').set_attr(
    mask=float(),
    pad_id=int(),
)
OpsRegister.Register('topk_avg_pooling').set_attr(
    top_ks=list(),
    feat_map_num=int(),
    is_pooling_by_row=bool(),
)
OpsRegister.Register('Dense').set_attr(
    axis=int(),
    out_dim=int(),
    bias_term=bool(),
)
OpsRegister.Register('MatchMatrix').set_attr(
    dim_in=int(),
    dim_t=int(),
    linear_term=bool(),
    bias_term=bool(),
    is_l_same=bool(),
)
