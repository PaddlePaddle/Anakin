
#include "saber/funcs/impl/bm/vender_conv.h"
#include "bmkernel_base.h"
#include "bm_common.h"
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "tensor_op.h"

namespace anakin
{
namespace saber
{

int get_align_tensor_size(bm_tensor_4d_t shape){
  int c_per_npu = ceiling_func_shift(shape.c, NPU_SHIFT);
  return shape.n * c_per_npu * get_neuron_csize_local(shape.h, shape.w);
}

void conv_splitc(bm_kernel_param_t kernel_param, conv_secs_info_t *secs_info){
  int oc_per_NPU = ceiling_func_shift(kernel_param.oc, NPU_SHIFT);
  int kernel_size = kernel_param.h * kernel_param.w * FLOAT_SIZE;
  int weight_capacity = kernel_param.ic * oc_per_NPU * kernel_size;
  secs_info->icsecs = 1;
  secs_info->ocsecs = 1;
  const int quart_local_size = (LOCAL_MEM_SIZE >> 2);
  if( weight_capacity > (LOCAL_MEM_SIZE >> 1) ){
    const int max_weight_size = quart_local_size;
    secs_info->icsecs = weight_capacity / max_weight_size + 1;
    if(secs_info->icsecs > kernel_param.ic){
      secs_info->icsecs = kernel_param.ic;
    }
    int icslice = (kernel_param.ic + secs_info->icsecs - 1) / secs_info->icsecs;
    weight_capacity = icslice * oc_per_NPU * kernel_size * FLOAT_SIZE;
    weight_capacity = addr_EU_align( weight_capacity);
    int max_ocsecs = oc_per_NPU;
    while( weight_capacity > max_weight_size ){
      if(secs_info->ocsecs == 1){
        secs_info->ocsecs = weight_capacity / quart_local_size + 1;
      }
      if(secs_info->ocsecs > max_ocsecs){
        secs_info->ocsecs = max_ocsecs;
        break;
      }else{
        secs_info->ocsecs++;
      }
      int ocslice = (kernel_param.oc + secs_info->ocsecs - 1) / secs_info->ocsecs;
      oc_per_NPU = ceiling_func_shift(ocslice, NPU_SHIFT);
      weight_capacity = icslice * oc_per_NPU * kernel_size * FLOAT_SIZE;
      weight_capacity = addr_EU_align(weight_capacity);
    }
  }
}

static bm_status_t conv_splith(bm_tensor_4d_t input_shape, bm_tensor_4d_t output_shape,
    bm_conv_param_t conv_param, int local_mem_capacity, int kh, conv_secs_info_t *secs_info){
  int io_need = get_align_tensor_size(input_shape) +
      get_align_tensor_size(output_shape);
  secs_info->hsecs = io_need / local_mem_capacity;
  int output_h = output_shape.h;
  output_shape.h = (output_h + secs_info->hsecs - 1) / secs_info->hsecs;
  input_shape.h = output_shape.h * conv_param.stride_h + kh;
  while(io_need > local_mem_capacity){
    if(secs_info->hsecs == output_h){
      return BM_NOT_SUPPORTED; 
    }
    secs_info->hsecs++;
    output_shape.h = (output_h + secs_info->hsecs - 1) / secs_info->hsecs;
    input_shape.h = output_shape.h * conv_param.stride_h + kh;
    io_need = get_align_tensor_size(input_shape) +
                       get_align_tensor_size(output_shape);
  }
  return BM_SUCCESS;
}

static bm_status_t get_conv_secs_info(
    bm_tensor_4d_t    input_shape,
    bm_kernel_param_t kernel_param,
    bm_tensor_4d_t    output_shape,
    bool              with_bias,
    bm_conv_param_t   conv_param,
    conv_secs_info_t *secs_info){
  int ic = kernel_param.ic;
  int oc = kernel_param.oc;
  int oc_per_NPU = ceiling_func_shift(oc, NPU_SHIFT);
  int bias_tensor_size = oc_per_NPU * FLOAT_SIZE;
  if(!with_bias){
    bias_tensor_size = 0;
  }
  int kernel_size = kernel_param.h * kernel_param.w;
  int weight_tensor_size = ic * oc_per_NPU * kernel_size * FLOAT_SIZE;
  int weight_capacity = addr_EU_align( weight_tensor_size  + bias_tensor_size);
  int ifmap_total_tensor_size = get_align_tensor_size(input_shape);
  int ofmap_total_tensor_size = get_align_tensor_size(output_shape);
  int totalneed_local_size = ifmap_total_tensor_size +
                          ofmap_total_tensor_size + weight_capacity;
  secs_info->nsecs = 1; secs_info->hsecs = 1;
  if(totalneed_local_size > LOCAL_MEM_SIZE){
    //if weight_capacity > 2 * bank_size then split oc and ic
    conv_splitc(kernel_param, secs_info);
    int ocslice = (oc + secs_info->ocsecs - 1) / secs_info->ocsecs;
    int icslice = (ic + secs_info->icsecs - 1) / secs_info->icsecs;
    oc_per_NPU = ceiling_func_shift(ocslice, NPU_SHIFT);

    weight_capacity = icslice * oc_per_NPU * kernel_size * FLOAT_SIZE;
    weight_capacity = addr_EU_align( weight_capacity + bias_tensor_size );
    int local_mem_capacity = LOCAL_MEM_SIZE - weight_capacity;
    CHECK_GT(local_mem_capacity, 0) << "local memory capacity not enough";
    input_shape.c = icslice;
    output_shape.c = ocslice;
    ifmap_total_tensor_size = get_align_tensor_size(input_shape);
    ofmap_total_tensor_size = get_align_tensor_size(output_shape);
    int totalneed_local_size = ifmap_total_tensor_size + ofmap_total_tensor_size;
    if(totalneed_local_size > local_mem_capacity){
      int kh_ext = conv_param.dilation_h * (kernel_param.h - 1) + 1;
      if(input_shape.n > 1){
        if( totalneed_local_size > local_mem_capacity * input_shape.n){
          secs_info->nsecs = input_shape.n;
          output_shape.n = input_shape.n = 1;
          bm_status_t result = conv_splith(input_shape, output_shape,
              conv_param, local_mem_capacity, kh_ext, secs_info);
          if(result == BM_NOT_SUPPORTED){
            return result;
          }
        }else{
          int input_n = input_shape.n;
          secs_info->nsecs = (totalneed_local_size + local_mem_capacity - 1) / local_mem_capacity;
          input_shape.n = (input_n + secs_info->nsecs - 1) / secs_info->nsecs;
          output_shape.n = input_shape.n;
          totalneed_local_size = get_align_tensor_size(input_shape) +
                       get_align_tensor_size(output_shape);
          while(totalneed_local_size > local_mem_capacity){
            secs_info->nsecs++;
            input_shape.n = (input_n + secs_info->nsecs - 1) / secs_info->nsecs;
            output_shape.n = input_shape.n;
            totalneed_local_size = get_align_tensor_size(input_shape) +
                       get_align_tensor_size(output_shape);
          }
        }
      }else{
        bm_status_t result = conv_splith(input_shape, output_shape,
            conv_param, local_mem_capacity, kh_ext, secs_info);
        if(result == BM_NOT_SUPPORTED){
          return result;
        }
      }
    }
  }else{
    secs_info->icsecs = 1;
    secs_info->ocsecs = 1;
  }
  return BM_SUCCESS;
}

// FP32 part
template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    create(const std::vector<Tensor<BM> *>& inputs,
            std::vector<Tensor<BM> *>& outputs,
            ConvParam<BM>& param, Context<BM>& ctx)
{
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    init(const std::vector<Tensor<BM> *> &inputs,
         std::vector<Tensor<BM> *> &outputs,
         ConvParam<BM> &param, Context<BM> &ctx)
{

    _handle = ctx.get_handle();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<BM>*>& inputs,
                std::vector<Tensor<BM>*>& outputs,
                ConvParam<BM>& param)
{
    const BM_mem_addr in_data = (const BM_mem_addr) inputs[0]->data();
    BM_mem_addr out_data = (BM_mem_addr) outputs[0]->mutable_data();
    const BM_mem_addr weight = (const BM_mem_addr) param.weight()->data();

    int input_n = inputs[0]->num();
    int input_c = inputs[0]->channel();
    int input_h = inputs[0]->height();
    int input_w = inputs[0]->width();

    int output_n = outputs[0]->num();
    int output_c = outputs[0]->channel();
    int output_h = outputs[0]->height();
    int output_w = outputs[0]->width();

    int group = param.group;
    int kh = param.weight()->height();
    int kw = param.weight()->width();
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int dilation_h = param.dilation_h;
    int dilation_w = param.dilation_w;

    bool with_bias = param.bias()->size() > 0;
    const bm_mem_desc bias = with_bias ? (const bm_mem_desc) param.bias()->data() : BM_MEM_NULL;

    bm_tensor_4d_t input_shape = {
        input_n,
        input_c,
        input_h,
        input_w};

    bm_tensor_4d_t output_shape = {
        output_n,
        output_c,
        output_h,
        output_w};

    bm_kernel_param_t kernel_param = {
        group,
        output_c,
        input_c,
        kh,
        kw};

    bm_conv_param_t conv_param = {
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        0};

    bm_device_mem_t input_buf_mem = in_data;
    // TODO: handle special case with pooling op

    conv_secs_info_t secs_info;
    bm_status_t result = get_conv_secs_info(input_shape, kernel_param,
          output_shape, with_bias, conv_param, &secs_info);
    CHECK_EQ(BM_SUCCESS, result) << "local memory is not enough in conv.";

    bm_api_conv_forward bm_conv_param = {
      bm_mem_get_device_addr(input_buf_mem),
      bm_mem_get_device_addr(out_data),
      bm_mem_get_device_addr(weight),
      with_bias ? bm_mem_get_device_addr(bias) : BM_MEM_ADDR_NULL,
      input_shape.n,
      input_shape.c,
      input_shape.h,
      input_shape.w,
      kernel_param.g,
      output_shape.c,
      kernel_param.h,
      kernel_param.w,
      conv_param.dilation_h,
      conv_param.dilation_w,
      conv_param.pad_h,
      conv_param.pad_w,
      conv_param.stride_h,
      conv_param.stride_w,
      with_bias,
      conv_param.result_add,
      secs_info.icsecs,
      secs_info.ocsecs,
      secs_info.nsecs,
      secs_info.hsecs
    };

    LOG(INFO)<<"BM Conv starts...";
    print_tensor(*inputs[0]);

    bm_status_t bm_stat = bmlib_kernel_launch(_handle, "/usr/local/include/bm/bmkernel_bin.bin");
    CHECK_EQ(BM_SUCCESS, bm_stat) << "bmlib_kernel_launch failed.";
    
    /* Send arguments. */
    enum BmOpType op = CONV;
    bmkernel_api_base api = { op, reinterpret_cast<void *>(&bm_conv_param) };
    BM_CHECK(bmlib_kernel_send_args(_handle, reinterpret_cast<void *>(&api), sizeof(api)));

    LOG(INFO)<<"BM Conv ends...";
    print_tensor(*outputs[0]);

    return SaberSuccess;
}

// INT8 part
// Not supported yet

template class VenderConv2D<BM, AK_FLOAT>;
} // namespace saber
} // namespace anakin