/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_CALL_CONF_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_CALL_CONF_H

#include <iostream>
#include <cstddef>
#include "saber/saber_types.h"
#include "stddef.h"

namespace anakin {
namespace saber {
namespace jit {

// convolution
enum conv_version_t {ver_unused, ver_fma, ver_avx512_core, ver_4fma, ver_4vnni, ver_vnni};
enum conv_loop_order_t { loop_cgn, loop_gnc, loop_ngc };
enum conv_kernel_kind_t {embd_bcast, expl_bcast};

enum conv_1x1_loop_order_t {
    loop_rbl, loop_rlb, loop_lbr, loop_lrb, loop_blr,
    loop_brl
};

enum {
    FLAG_MB_FIRST = 1 << 0, FLAG_MB_LAST = 1 << 1,
    FLAG_OC_FIRST = 1 << 2, FLAG_OC_LAST = 1 << 3,
    FLAG_IC_FIRST = 1 << 4, FLAG_IC_LAST = 1 << 5,
    FLAG_SP_FIRST = 1 << 6, FLAG_SP_LAST = 1 << 7,
    FLAG_REDUCE_FIRST = 1 << 8, FLAG_REDUCE_LAST = 1 << 9,
};

struct jit_int8_packed_fc_call_t {
    const void *src{nullptr};
    const void *weights{nullptr};
    const void *output_data{nullptr};

    size_t lda{0}; // used in backward_weights only
    size_t ldb{0};
    size_t ldc{0};
    size_t k_block{0};

};

struct jit_int8_packed_fc_config_t {
    size_t m_block_size{0};
    size_t n_block_size{0};
    size_t k_block_number{0};
};


struct jit_1x1_conv_call_t {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data; // used in forward and backward_weights only
    const void *acc_s32;
    const void *scales;
    const void *compensation;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;
    size_t output_stride; // used in backward_weights only
    size_t first_last_flag;
    size_t reduce_pos_flag;
};

struct jit_conv_call_t {
    const void *src{nullptr}; /* hack, non-const for backward_data */
    const void *dst{nullptr}; /* hack, non-const for forward */
    const void *filt{nullptr}; /* hack, non-const for backward_weights */
    const void *bias{nullptr}; /* hack, non-const for backward_bias */
    const void *src_prf{nullptr};
    const void *dst_prf{nullptr};
    const void *filt_prf{nullptr};
    const void *bias_prf{nullptr};
    const void *scales{nullptr};
    const void *acc_s32{nullptr};
    const void *compensation{nullptr};
    size_t kd_padding{0};
    size_t kd_padding_prf{0};
    size_t kh_padding{0};
    size_t kh_padding_prf{0};
    size_t kw_padding{0};
    size_t channel{0};
    size_t channel_prf{0};
    size_t oc_blocks{0};
    size_t ur_w{0};
    size_t ur_str_w{0};
    size_t ch_blocks{0};
    size_t t_overflow{0};
    size_t b_overflow{0};
    int flags{0};
};

struct jit_wino_transform_call_s {
    size_t tile_block;
    size_t tile_block_ur;
    size_t nb_tile_block_ur;
    size_t tile_count;
    size_t tj;
    size_t ti;
    void *src;
    void *dst;
    void *Mw;
    void *M;
    void *T;
    void *G;
    void *bias;
};

struct jit_conv_conf_t {
    conv_version_t ver{ver_unused};
    conv_loop_order_t loop_order{loop_cgn};
    LayoutType src_fmt{Layout_invalid};
    int ndims{0};
    int mb{0};
    int ngroups{0};
    int ic{0};
    int oc{0};
    int oc_without_padding{0};
    int ic_without_padding{0};
    int id{0};
    int ih{0};
    int iw{0};
    int od{0};
    int oh{0};
    int ow{0};
    int f_pad{0};
    int l_pad{0};
    int t_pad{0};
    int back_pad{0};
    int r_pad{0};
    int b_pad{0};
    int kd{0};
    int kh{0};
    int kw{0};
    int stride_d{0};
    int stride_h{0};
    int stride_w{0};
    int dilate_d{0};
    int dilate_h{0};
    int dilate_w{0};
    bool with_bias{false};
    bool with_relu{false};
    float relu_negative_slope{0.f};
    bool with_sum{false};
    bool is_dw{false};
    bool is_dw_int8{false};
    int idp{0};
    int ihp{0};
    int iwp{0};
    int ohp{0};
    int owp{0};
    int nb_ic{0};
    int ic_block{0};
    int nb_oc{0};
    int oc_block{0};
    int nb_g{0};
    int g_block{0};
    int nb_ic_blocking{0};
    int nb_oc_blocking{0}; // blocking of nb_ic and nb_i{0c
    int nb_ic_blocking_max{0};
    int nb_ic_L2{0};
    int nb_oc_L2{0};
    int ur_h{0};
    int ur_w{0};
    int ur_w_tail{0};
    bool is_1stconv{0};
    /* fma avx512_core */
    conv_kernel_kind_t kernel_kind{embd_bcast};
    /* 4fma */
    int tr_iw{0};
    int tr_src_num_guard_elems{0};
    /* 1st conv: 4fma */
    int tr_ld{0};
    int kh_step{0};
    /* 4vnni */
    int typesize_in{0};
    int typesize_out{0};
    int typesize_bia{0};
    int typesize_acc{0};
    int tr_ow{0};
    /* avx512_u8s8u8 */
    int ic_nb1{0};
    int ic_nb2{0};
    int oc_nb1{0};
    int ur_ow_max{0};
    int ur_ow{0};
    int ur_ow_tail{0};
    int ur_ow_nsteps{0};
    DataType bia_dt{AK_INVALID};
    DataType dst_dt{AK_INVALID};
    DataType sum_dt{AK_INVALID};
    /* avx512: max possible value is nregs(32) - aux_regs(4) */
    int src_offsets[28]{0};
    int src_count{0};
    bool expl_bcast{false};
    bool large_spatial{false};
    int is_oc_scale{0};
    bool signed_input{false};
    float wei_adj_scale{0.f};

    // gemm conv
    int is{0};
    int os{0};
    int ks{0};
    ptrdiff_t im2col_sz;
    bool need_im2col{false};
    int nthr{0};

    // dw conv
    int nb_ch{0};
    int ch_block{0};
    int nb_ch_blocking{0};
    round_mode rm{nearest};

    // pooling
    bool with_partial_pool=false;
    PoolingType pool_alg{Pooling_unknow};
    int pool_kw{0};

    //the scale for post sum
    float sum_scale{0.f};

    // output layout nhwc
    bool output_nhwc{false};
};

struct jit_1x1_conv_conf_t {
    conv_version_t ver{ver_unused};

    int mb{0};
    int ngroups{0};
    int ic{0};
    int oc{0};
    int oc_without_padding{0};
    int ic_without_padding{0};
    int iw{0};
    int ih{0};
    int ow{0};
    int oh{0};
    int l_pad{0};
    int t_pad{0};
    int kh{0};
    int kw{0};
    int stride_h{0};
    int stride_w{0};
    bool with_bias{false};
    bool with_relu{false};
    float relu_negative_slope{0.f};
    bool with_sum{false};

    int is{0}; 
    int os{0};
    int ic_block{0};
    int oc_block{0};

    int ur{0};
    int ur_tail{0};

    int reduce_dim{0};
    int reduce_block{0};
    int nb_reduce{0};
    int nb_reduce_blocking{0};
    int nb_reduce_blocking_max{0};
    int load_dim{0};
    int load_block{0};
    int nb_load{0};
    int nb_load_blocking{0};
    int nb_load_blocking_max{0};
    int bcast_dim{0};
    int bcast_block{0};
    int nb_bcast{0};
    int nb_bcast_blocking{0};
    int nb_bcast_blocking_max{0};

    int reduce_loop_unroll{0};
    int reduce_loop_bcast_step{0};
    int reduce_loop_load_step{0};
    int load_loop_load_step{0};
    int load_loop_iter_step{0};
    int bcast_loop_output_step{0};
    int bcast_loop_output_substep{0};
    int bcast_loop_bcast_step{0};
    int bcast_loop_bcast_substep{0};
    int fma_step{0};
    int load_grp_count{0};
    conv_1x1_loop_order_t loop_order{loop_rbl};
    bool use_vmovntps{false};
    /* avx512 core */
    bool expl_bcast{false};
    /* 4vnni */
    int typesize_in{0};
    int typesize_out{0};
    int typesize_bia{0};
    int typesize_acc{0};
    /* 4fma */
    bool transpose_src{false};
    int tr_is{0};
    int nthr{0};
    int nthr_mb{0};
    int nthr_g{0};
    int nthr_oc_b{0};
    int nthr_ic_b{0};
    int is_oc_scale{0};
    DataType bia_dt{AK_INVALID};
    DataType src_dt{AK_INVALID};
    DataType dst_dt{AK_INVALID};
    DataType sum_dt{AK_INVALID};
    round_mode rm{nearest};
    bool signed_input{false};
    float wei_adj_scale{0.f};

    //the scale for post sum
    float sum_scale{0.f};
};

struct jit_conv_conf_2x3_wino_t {
    conv_version_t ver;

    int m;
    int r;
    int alpha;
    int tile_h, tile_w;

    int mb;
    int ngroups, ic, oc, oc_without_padding;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;

    int nb_ic, ic_block;
    int nb_oc, oc_block;

    int w_block_size, h_block_size;

    DataType bia_dt;
    DataType dst_dt;

    int is_oc_scale;
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;

    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;
    bool small_mb;

    int xb, yb;
    int inp_stride;
    int out_stride;
    int wei_stride;
    int bia_stride;

    int M, N, K;
    int m_block, n_block, k_block;
    int n2_block, n_chunks;
    int k2_block, k_chunks;

    round_mode rm;
    float sum_scale;
};


// pooling
struct jit_pool_conf_t {
    int ndims{0};
    int mb{0};
    int c{0};
    int id{0};
    int ih{0};
    int iw{0};
    int od{0};
    int oh{0};
    int ow{0};
    int stride_d{0};
    int stride_h{0};
    int stride_w{0};
    int kd{0};
    int kh{0};
    int kw{0};
    int f_pad{0};
    int t_pad{0};
    int l_pad{0};
    PoolingType alg{Pooling_unknow};
    bool pad_w_is_null{0};
    bool simple_alg{0};
    DataType ind_dt{AK_INVALID};
    LayoutType src_fmt{Layout_invalid};

    int c_block{0};
    int c_tail{0};
    int nb_c{0};
    int ur_c{0};
    int ur_c_tail{0};
    int ur_w{0};
    int ur_w_tail{0};
    size_t tail[4]{0,0,0,0};
    DataType src_dt{AK_INVALID};
    DataType dst_dt{AK_INVALID};
};

struct jit_pool_call_t {
    const float *src;
    const float *dst;
    const void *indices;
    const float *src_prf;
    const float *dst_prf;
    const void *indices_prf;
    size_t oh;
    size_t kd_padding;
    size_t kh_padding;
    size_t kh_padding_shift;
    size_t kd_padding_shift;
    size_t kw_padding;
    const float* init_value;
    float ker_area_h;
};

struct jit_pool_call_nhwc_t {
    union {
        const unsigned char *src_i8;
        const float *src_fp32;
    };
    union {
        unsigned char *dst_i8;
        float *dst_fp32;
    };
    /*
       valid kernel range of pooling operation,
       could be different with kw/kh if has padding.
    */
    size_t kw_range;
    size_t kh_range;
    /* idivider is 1/(kw_range*kh_range) or 1/(kw*kh) */
    float  idivider;
};

// concat with optional relu fusion
struct jit_concat_call_t {
  const unsigned char** src;
  const int  *nb_ic;
  const unsigned char* dst;
  const float *scale;
};

struct jit_concat_conf_t {
  int           bs;
  int           h, w;
  int           oc;
  int           n_inputs;
  int           typesize;
  unsigned int  *block;      // u8: 64, s32: 16
  int           bits_size;  // 128, 256, 512 : xmm, ymm, zmm
  bool          with_relu;
  DataType      src_dt;
  DataType      dst_dt;
  float         scale_max;
  float         *scales;
  unsigned int  *nb_ic;
  unsigned int  *ic;
  unsigned long *tail;
};

struct jit_axpy_call_t {
  const void **src;
  const void *dst;
  size_t work_amount;
};

struct jit_axpy_conf_t {
  int           n_inputs;
  int           bs;
  int           h, w;
  int           oc;
  int           n;
  DataType      dt;
  int           typesize;
  int           block_size;      // u8: 64, s32: 16
  int           bits_size;  // 128, 256, 512 : xmm, ymm, zmm
};

struct jit_eltwise_call_t {
  const void **src;
  const void *dst;
  size_t work_amount;
};

struct jit_eltwise_conf_t {
  int           n_inputs;
  DataType      dt;
  int           typesize;
  bool          with_relu;
  const float   *scales;
};

struct jit_priorbox_call_t{
  const void *dst;
  float start;
  const void *start_offset;
  float offset;
  float step;
  float box_length;
  float img_length;
  size_t work_amount;
  float block = 8.0f;
};

struct jit_priorbox_conf_t{
  bool is_add;
};

// gemm conv
struct jit_gemm_deconv_conf_t {
    int mb;
    int ic, ih, iw, oc, oh, ow;
    int stride_h, stride_w;
    int kh, kw;
    int f_pad, t_pad, l_pad;
    int dilate_d, dilate_h, dilate_w;
};

struct jit_deconv_conf_t {
    conv_version_t ver{ver_unused};
    LayoutType src_fmt{Layout_invalid};
    int ndims{0};
    int mb{0};
    int ngroups{0};
    int ic{0};
    int oc{0};
    int oc_without_padding{0};
    int ic_without_padding{0};
    int ih{0};
    int iw{0};
    int oh{0};
    int ow{0};
    int l_pad{0};
    int t_pad{0};
    int back_pad{0};
    int r_pad{0};
    int b_pad{0};
    int kh{0};
    int kw{0};
    int stride_h{0};
    int stride_w{0};
    int dilate_h{0};
    int dilate_w{0};
    bool with_bias{false};
    bool with_relu{false};
    float relu_negative_slope{0.f};
    bool with_sum{false};
    int nb_ic{0};
    int ic_block{0};
    int nb_oc{0};
    int oc_block{0};
    int nb_g{0};
    int g_block{0};
    int nb_ic_blocking{0};
    int nb_oc_blocking{0}; // blocking of nb_ic and nb_ic
    int nb_ic_blocking_max{0};
    int nb_ic_L2{0};
    int nb_oc_L2{0};
    int ur_h{0};
    int ur_w{0};
    int ur_w_tail{0};
    int typesize_in{0};
    int typesize_out{0};

    /* fma avx512_core */
    conv_kernel_kind_t kernel_kind{embd_bcast};
};

struct jit_deconv_call_t {
    const void *src{nullptr}; /* hack, non-const for backward_data */
    const void *dst{nullptr}; /* hack, non-const for forward */
    const void *filt{nullptr}; /* hack, non-const for backward_weights */
    const void *bias{nullptr}; /* hack, non-const for backward_bias */
    const void *src_prf{nullptr};
    const void *dst_prf{nullptr};
    const void *filt_prf{nullptr};
    const void *bias_prf{nullptr};
    const void *scales{nullptr};
    size_t kh_padding{0};
    size_t kh_padding_prf{0};
    size_t channel{0};
    size_t channel_prf{0};
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif
