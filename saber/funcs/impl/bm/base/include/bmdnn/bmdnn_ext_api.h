#ifndef BMDNN_EXT_API_H
#define BMDNN_EXT_API_H

#include "bmdnn_runtime.h"

#if defined (__cplusplus)
extern "C" {
#endif

bm_status_t bmdnn_threshold_forward(
    bm_handle_t      handle,
    float               threshold,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output
    );

bm_status_t bmdnn_exp_forward(
    bm_handle_t      handle,
    float               base,
    float               input_scale,
    float               input_shift,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output
    );

bm_status_t bmdnn_exp_backward(
    bm_handle_t      handle,
    float               base,
    float               input_scale,
    float               input_shift,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff
    );

bm_status_t bmdnn_power_forward(
    bm_handle_t      handle,
    float               power_,
    float               scale_,
    float               shift_,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output
    );

bm_status_t bmdnn_power_backward(
    bm_handle_t      handle,
    float               power_,
    float               scale_,
    float               shift_,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff
    );

bm_status_t bmdnn_euclidean_loss_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  label,
    bm_device_mem_t  temp_,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  diff,
    bm_device_mem_t  loss);

bm_status_t bmdnn_euclidean_loss_backward(
    bm_handle_t      handle,
    float               alpha,
    //input
    bm_device_mem_t  output,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_silence_backward(
    bm_handle_t      handle,
    //input
    //bm_device_mem_t  output_data,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_lstm_unit_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  X_i,
    bm_device_mem_t  X_f,
    bm_device_mem_t  X_o,
    bm_device_mem_t  X_g,
    bm_device_mem_t  C_prev,
    bm_device_mem_t  cont_expand,
    int                 num,
    int                 hidden_dim,
    //output
    bm_device_mem_t  C,
    bm_device_mem_t  H);

bm_status_t bmdnn_lstm_unit_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  C_diff,
    bm_device_mem_t  H_diff,
    bm_device_mem_t  X_i,
    bm_device_mem_t  X_f,
    bm_device_mem_t  X_o,
    bm_device_mem_t  X_g,
    bm_device_mem_t  C_prev,
    bm_device_mem_t  C,
    bm_device_mem_t  cont_expand,
    int                 num,
    int                 hidden_dim,
    //output
    bm_device_mem_t  C_prev_diff,
    bm_device_mem_t  X_i_diff,
    bm_device_mem_t  X_f_diff,
    bm_device_mem_t  X_o_diff,
    bm_device_mem_t  X_g_diff);

bm_status_t bmdnn_eltwise_forward(
    bm_handle_t      handle,
    int                 op_,
    int                 flag_first,
    float               coeffs_,
    int                 index,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  target,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  mask_data,
    bm_device_mem_t  output);

bm_status_t bmdnn_eltwise_backward(
    bm_handle_t      handle,
    int                 op_,
    int                 flag_first,
    float               coeffs_,
    int                 index,
    //input
    bm_device_mem_t  output_data,
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input_data,
    bm_device_mem_t  mask_data,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_bias_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  bias,
    int                 outer_dim,
    int                 dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_bias_backward(
    bm_handle_t      handle,
    int                 flag,
    //input
    bm_device_mem_t  output_diff,
    int                 outer_dim,
    int                 bias_dim,
    int                 inner_dim,
    //output
    bm_device_mem_t  input_diff,
    bm_device_mem_t  bias_diff);

bm_status_t bmdnn_log_forward(
    bm_handle_t      handle,
    float               scale,
    float               shift,
    float               base,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_log_backward(
    bm_handle_t      handle,
    float               scale,
    float               shift,
    float               base,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_absval_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_absval_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_sigmoid_cross_entropy_loss_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  target,
    bm_device_mem_t  buffer,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output,
    bm_device_mem_t  loss);

bm_status_t bmdnn_sigmoid_cross_entropy_loss_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output,
    bm_device_mem_t  target,
    bm_device_mem_t  output_diff,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_contrastive_loss_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input_0,
    bm_device_mem_t  input_1,
    bm_device_mem_t  label,
    bm_device_mem_t  buffer,
    int                 input_n,
    int                 input_c,
    float               margin,
    bool                legacy_version,
    //output
    bm_device_mem_t  diff,
    bm_device_mem_t  dist_sq,
    bm_device_mem_t  loss);

bm_status_t bmdnn_contrastive_loss_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  label,
    bm_device_mem_t  diff,
    bm_device_mem_t  dist_sq,
    bm_device_mem_t  output_diff,
    bm_device_mem_t  buffer,
    int                 input_n,
    int                 input_dim,
    float               margin,
    bool                legacy_version,
    int                 propagate_down_flag,
    //output
    bm_device_mem_t  input_diff_0,
    bm_device_mem_t  input_diff_1);

bm_status_t bmdnn_filter_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  filter,
    int                 input_n,
    int                 output_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_filter_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  filter,
    int                 input_n,
    int                 output_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_split_backward(
    bm_handle_t      handle,
    //input
    int                 is_first,
    bm_device_mem_t  output_diff,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_bnll_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_bnll_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    float               threshold,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_prelu_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  slope,
    float            slope0,
    int                 channel_shared,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_prelu_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    bm_device_mem_t  slope,
    int                 propagate_down_flag,
    int                 channel_shared,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  slope_diff,
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_scale_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  scale,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 scale_dim,
    int                 inner_dim,
    int                 scale_is_neuron,
    //output
    bm_device_mem_t  scale_extension,
    bm_device_mem_t  output);

bm_status_t bmdnn_scale_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input_data,
    bm_device_mem_t  scale_extension,
    int                 propagate_down_flag,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 scale_dim,
    int                 inner_dim,
    int                 scale_is_neuron,
    //output
    bm_device_mem_t  scale_diff,
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_elu_forward(
    bm_handle_t      handle,
    float               alpha,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_elu_backward(
    bm_handle_t      handle,
    float               alpha,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

#if defined (__cplusplus)
}
#endif

#endif /* BMDNN_EXT_API_H */
