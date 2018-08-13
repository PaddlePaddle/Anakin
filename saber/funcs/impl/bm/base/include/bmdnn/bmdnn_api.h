#ifndef BMDNN_API_H
#define BMDNN_API_H

#include "bmdnn_runtime.h"
#include "op_code.h"

#if defined (__cplusplus)
extern "C" {
#endif

/*
 * All the name-style of input/output are in the viewpoint of forward operation
 */

typedef struct kernel_param{
    int g;
    int oc;
    int ic;
    int h;
    int w;
}bm_kernel_param_t;

typedef struct bm_conv_param{
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    bool result_add;
}bm_conv_param_t;

typedef struct bm_pool_param{
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int kh;
  int kw;
  bool is_avg_pooling;
}bm_pool_param_t;

bm_status_t bmdnn_conv_relu_pool_forward(
    bm_handle_t      handle,
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    bm_device_mem_t  bias,
    bm_tensor_4d_t      input_shape,
    bm_kernel_param_t   kernel_param,
    bm_pool_param_t     pool_param,
    bm_conv_param_t     conv_param,
    bool                with_bias,
    bm_device_mem_t  output);

bm_status_t bmdnn_conv_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    bm_device_mem_t  bias,
    bm_tensor_4d_t      input_shape,
    bm_kernel_param_t   kernel_param,
    bm_tensor_4d_t      output_shape,
    bm_conv_param_t     conv_param,
    bool                with_bias,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_deconv_forward(
    bm_handle_t      handle,
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    bm_device_mem_t  bias,
    bm_tensor_4d_t      input_shape,
    bm_kernel_param_t   kernel_param,
    bm_tensor_4d_t      output_shape,
    bm_conv_param_t     conv_param,
    bool                with_bias,
    bm_device_mem_t  output);

bm_status_t bmdnn_conv_backward_bias(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 groups,
    int                 output_c,
    int                 kh,
    int                 kw,
    int                 pad_h,
    int                 pad_w,
    int                 stride_h,
    int                 stride_w,
    int                 result_add,
    //output
    bm_device_mem_t  bias_diff);

bm_status_t bmdnn_pooling_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 kh,
    int                 kw,
    int                 pad_h,
    int                 pad_w,
    int                 stride_h,
    int                 stride_w,
    int                 is_avg_pooling,
    //output
    bm_device_mem_t  output
    );
bm_status_t bmdnn_upsample_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 size,
    //output
    bm_device_mem_t  output
    );
bm_status_t bmdnn_roi_pooling_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  rois,
    int              input_n,
    int              input_c,
    int              input_h,
    int              input_w,
    int              pooled_h,
    int              pooled_w,
    int              roi_num,
    int              spatial_scale,
    //output
    bm_device_mem_t  output
    );

bm_status_t bmdnn_fc_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    bm_device_mem_t  bias,
    int              batch_size,
    int              num_output_neuron,
    int              num_input_neuron,
    int              transpose,
    int              using_bias,
    int              using_relu,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_fc_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    int              num_output_neuron,
    int              batch_size,
    int              num_input_neuron,
    int              using_bias,
    int              propagate_down_bias_diff,
    int              propagate_down_weight_diff,
    int              propagate_down_bottom,
    //output
    bm_device_mem_t  weight_diff,
    bm_device_mem_t  bias_diff,
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_dropout_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    float            dropout_ratio,
    int              input_n,
    int              input_dim,
    //output
    bm_device_mem_t  output,
    bm_device_mem_t  mask);

bm_status_t bmdnn_dropout_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    float               dropout_ratio,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_batchnorm_forward_inference(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  mean_ma,
    bm_device_mem_t  variance_ma,
    float               scale_ma,
    bm_device_mem_t  variance,
    float               eps,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_batchnorm_forward_train(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    float               ma_fraction,
    float               eps,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  output,
    bm_device_mem_t  mean,
    bm_device_mem_t  variance,
    bm_device_mem_t  mean_ma,
    bm_device_mem_t  variance_ma);

bm_status_t bmdnn_batchnorm_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    bm_device_mem_t  variance,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 using_global_stats,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_lrn_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 lrn_n,
    float               alpha,
    float               beta,
    float               k,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_lrn_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    bm_device_mem_t  input,
    int                 lrn_n,
    float               alpha,
    float               beta,
    float               k,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_relu_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    float               negative_slope,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_relu_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    float               negative_slope,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_sigmoid_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_sigmoid_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_tanh_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_tanh_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    int                 input_n,
    int                 input_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_softmax_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_inner_dim,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_softmax_backward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  output,
    int                 input_n,
    int                 input_c,
    int                 input_inner_dim,
    //output
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_softmax_loss_forward(
    bm_handle_t      handle,
    bm_device_mem_t  input,
    bm_device_mem_t  label,
    float               normalizer,
    int                 input_n,
    int                 input_c,
    int                 input_inner_dim,
    bm_device_mem_t  output,
    bm_device_mem_t  loss);
bm_status_t bmdnn_interp_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 pad_bag,
    int                 pad_end,
    int                 output_h,
    int                 output_w,
    int                 platform_sp,
    //output
    bm_device_mem_t  output
    );
bm_status_t bmdnn_softmax_loss_backward(
    bm_handle_t      handle,
    bm_device_mem_t  output,
    bm_device_mem_t  label,
    bm_device_mem_t  loss,
    float               normalizer,
    int                 input_n,
    int                 input_c,
    int                 input_inner_dim,
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_softmax_loss_bidirection(
    bm_handle_t      handle,
    bm_device_mem_t  input,
    bm_device_mem_t  label,
    float               normalizer,
    int                 input_n,
    int                 input_c,
    int                 input_inner_dim,
    bm_device_mem_t  output_diff,
    bm_device_mem_t  loss);

bm_status_t bmdnn_multiregion_forward_parallel(
    bm_handle_t         handle,
    //input
    bm_device_mem_t*     input,
    int*                 input_n,
    int*                 input_c,
    int*                 input_h,
    int*                 input_w,
    int                  input_num,
    int                 classes,
    int                 coords,
    int                 nums,
    int*                 Activate_parm,
    //output
    bm_device_mem_t*  output
);

bm_status_t bmdnn_accuracy(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  label_idx,
    bm_device_mem_t  input_mem_buffer,
    int                 input_num,
    int                 input_dim,
    int                 top_k,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_coeff_update_sgd(
    bm_handle_t      handle,
    bm_device_mem_t  weight_diff,
    bm_device_mem_t  weight,
    bm_device_mem_t  history_weight,
    int                 weight_count,
    float               base_lr,
    float               momentum,
    float               weight_decay);

bm_status_t bmdnn_fc_backward_sgd(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  output_diff,
    bm_device_mem_t  input,
    //input and output
    bm_device_mem_t  weight,
    bm_device_mem_t  weight_history,
    int                 num_output_neuron,
    int                 batch_size,
    int                 num_input_neuron,
    int                 using_bias,
    int                 propagate_down_bias_diff,
    int                 propagate_down_weight_diff,
    int                 propagate_down_bottom,
    float               base_lr,
    float               momentum,
    float               weight_decay,
    //output
    bm_device_mem_t  bias_diff,
    bm_device_mem_t  input_diff);

bm_status_t bmdnn_permute(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  output);

bm_status_t bmdnn_normalize_forward(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  scale,
    float               eps,
    float               scale_val,
    bool                across_spatial,
    bool                channel_share,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    //output
    bm_device_mem_t  output);

/*
 * MD Operations for user
 */


bm_status_t bmdnn_md_scalar(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    bm_device_mem_t  tensor_B,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    ALIGN_TENSOR_OP             align_tensor_op,
    int                 result_add,
    int                 A_is_constant,
    int                 B_is_constant,
    float               A_const_val,
    float               B_const_val,
    int                 B_N_is_1,
    int                 B_index_is_1,
    //output
    bm_device_mem_t  tensor_R);

bm_status_t bmdnn_md_cmp(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    bm_device_mem_t  tensor_B,
    bm_device_mem_t  tensor_C,
    bm_device_mem_t  tensor_D,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 A_is_constant,
    int                 B_is_constant,
    int                 C_is_constant,
    int                 D_is_constant,
    float               A_constant,
    float               B_constant,
    unsigned int        C_constant,
    unsigned int        D_constant,
    int                 result_skip,
    //output
    bm_device_mem_t  tensor_Y,
    bm_device_mem_t  tensor_R);

bm_status_t bmdnn_md_sfu(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    SFU_OP              sfu_op,
    float               a,
    int                 n,
    //output
    bm_device_mem_t  tensor_Y);

bm_status_t bmdnn_md_sum(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 result_add,
    //output
    bm_device_mem_t  tensor_Y);


bm_status_t bmdnn_md_linear(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    bm_device_mem_t  tensor_B,
    bm_device_mem_t  tensor_S,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    LINEAR_OP           linear_op,
    int                 result_add,
    int                 B_is_const,
    int                 S_is_const,
    float               B_const_val,
    float               S_const_val,
    //output
    bm_device_mem_t  tensor_Y);

bm_status_t bmdnn_img_sum(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  tensor_A,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 result_add,
    //output
    bm_device_mem_t  tensor_Y);

/*
 * fullnet mode
 */
bm_status_t bmdnn_fullnet(
        bm_handle_t handle,
        unsigned long long bdc_cmd_offset,
        unsigned long long gdma_cmd_offset,
        unsigned long long cdma_cmd_offset,
        unsigned long long cmd_num_offset
        );

/*
 * multiple fullnet mode
 */
bm_status_t bmdnn_multi_fullnet(
        bm_handle_t handle,
        int input_num,
        unsigned long long* user_input_global_offset,
        unsigned long long* cmd_input_global_offset,
        int* input_tensor_size,
        int output_num,
        unsigned long long* user_output_global_offset,
        unsigned long long* cmd_output_global_offset,
        int* output_tensor_size,
        unsigned long long bdc_cmd_offset,
        unsigned long long gdma_cmd_offset,
        unsigned long long cdma_cmd_offset,
        int* bdc_cmd_num,
        int* gdma_cmd_num,
        int* cdma_cmd_num,
        int cmdgroup_num
        );

/*
 * dynamic fullnet mode
 */
bm_status_t bmdnn_dynamic_fullnet(
        bm_handle_t handle,
        unsigned long long compiled_ir_global_addr,
        unsigned int compiled_ir_length,
        unsigned int batch_num,
        unsigned int input_num,
        unsigned long long* input_global_offset,
        unsigned int* input_height,
        unsigned int* input_width,
        unsigned int output_num,
        unsigned long long* output_global_offset,
        unsigned long long apd_ctx_mem_offset
#if defined(USING_CMODEL) && !defined(USING_FULLNET)
        ,float**    p_refer_result
#endif
        );

/**
  * Depthwise convolution.
  */
bm_status_t bmdnn_depthwise_forward(
        bm_handle_t         handle,
        bm_device_mem_t     input,
        bm_device_mem_t     weight,
        bm_device_mem_t     bias,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 kernel_h,
        int                 kernel_w,
        int                 dilation_h,
        int                 dilation_w,
        int                 pad_h,
        int                 pad_w,
        int                 stride_h,
        int                 stride_w,
        int                 using_bias,
        bm_device_mem_t     output);

bm_status_t bmdnn_lstm_forward(
        bm_handle_t      handle,
        //input
        bm_device_mem_t  input,
        bm_device_mem_t  cont,
        bm_device_mem_t  input_static,
        /*bm_device_mem_t  w_hc,
        bm_device_mem_t  w_xc,*/
        bm_device_mem_t  w_hxc,
        bm_device_mem_t  w_xstatic,
        bm_device_mem_t  b_c,
        bm_device_mem_t  h_0,
        bm_device_mem_t  c_0,
        int                 input_n,
        int                 seq_len,
        int                 input_dim,
        int                 input_static_dim,
        int                 output_dim,
        int                 with_input_static,
        //output
        bm_device_mem_t  c,
        bm_device_mem_t  gate,
        bm_device_mem_t  h_T,
        bm_device_mem_t  c_T,
        bm_device_mem_t  h);

bm_status_t bmdnn_netease_ocr_forward(
        bm_handle_t      handle,
        //input
        bm_device_mem_t  conv1_ifmap,
        bm_device_mem_t  params,
        bm_device_mem_t  result);

typedef struct dim4_s {
    int n, c, h, w;
} dim4_t;
enum
{
    CONV_DEPTHWISE,
    CONV_3D
};
typedef struct mobilenet_conv_param_s
{
    /** convolution. */
    int type;
    bm_device_mem_t kernel;
    bm_device_mem_t bias;
    dim4_t          kernel_shape;
    int             dilation_h, dilation_w;
    int             pad_h, pad_w;
    int             stride_h, stride_w;
    bool            using_bias;
    /** batchnorm. */
    bm_device_mem_t mean;
    bm_device_mem_t variance;
    /** relu. */
    float           slope;
} mobilenet_conv_param_t;
bm_status_t bmdnn_mobilenet_forward(
        bm_handle_t handle,
        const mobilenet_conv_param_t   *conv,
        int                             num,
        const dim4_t                   &input_shape,
        const bm_device_mem_t          &input_global_mem,
        dim4_t                         &output_shape,
        bm_device_mem_t                &output_global_mem,
        float                           parallel_performance_factor = 1.f);

bm_status_t bmdnn_conv_forward_bank_conflict(
    bm_handle_t         handle,
    //input
    bm_device_mem_t     input,
    bm_device_mem_t     weight,
    bm_device_mem_t     bias,
    bm_tensor_4d_t      input_shape,
    bm_kernel_param_t   kernel_param,
    bm_tensor_4d_t      output_shape,
    bm_conv_param_t     conv_param,
    bool                with_bias,
    //output
    bm_device_mem_t     output);

bm_status_t bmdnn_pooling_forward_bank_conflict(
    bm_handle_t         handle,
    //input
    bm_device_mem_t     input,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 kh,
    int                 kw,
    int                 pad_h,
    int                 pad_w,
    int                 stride_h,
    int                 stride_w,
    int                 is_avg_pooling,
    bm_device_mem_t  output);

bm_status_t bmdnn_fc_forward_bank_conflict(
    bm_handle_t      handle,
    //input
    bm_device_mem_t  input,
    bm_device_mem_t  weight,
    bm_device_mem_t  bias,
    int              batch_size,
    int              num_output_neuron,
    int              num_input_neuron,
    int              transpose,
    int              using_bias,
    int              using_relu,
    bm_device_mem_t  output);

bm_status_t bmdnn_conv_forward_power_evaluation(
    bm_handle_t         handle,
    //input
    bm_device_mem_t     input,
    bm_device_mem_t     weight,
    bm_device_mem_t     bias,
    bm_tensor_4d_t      input_shape,
    bm_kernel_param_t   kernel_param,
    bm_tensor_4d_t      output_shape,
    bm_conv_param_t     conv_param,
    bool                with_bias,
    //output
    bm_device_mem_t     output);

bm_status_t bmdnn_img_scale(
        bm_handle_t handle, bm_device_mem_t dst, bm_device_mem_t src, int n,
        int c, int dh, int sh, int dw, int sw);

bm_status_t bmdnn_bn_forward_inference(
    bm_handle_t      handle,
    bm_device_mem_t  input,
    bm_device_mem_t  output,
    bm_device_mem_t  mean_ma,
    bm_device_mem_t  variance_ma,
    bm_device_mem_t  scale,
    bm_device_mem_t  bias,
    bm_device_mem_t  scale_ext,              
    float            eps,
    int              input_n,
    int              input_c,
    int              input_h,
    int              input_w
  );
#if defined (__cplusplus)
}
#endif

#endif /* BMDNN_API_H */
