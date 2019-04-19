#include "saber/funcs/impl/cuda/saber_anchor_generator.h"
#include "saber/core/tensor_op.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_anchor_generator_fwd(Dtype * out_data, \
                    Dtype* var_data,
                    const Dtype* in_data,
                    const int in_h, 
                    const int in_w, 
                    const float* anchor_sizes_data,
                    const int anchor_sizes_size, 
                    const float* aspect_ratios_data,
                    const int aspect_ratios_size,
                    const int num_anchors,
                    const int stride_h,
                    const int stride_w,
                    const float var_0,
                    const float var_1,
                    const float var_2,
                    const float var_3,
                    const float offset,
                    const int count)
{
    CUDA_KERNEL_LOOP(tid, count){
        int h_id = tid / (num_anchors * in_w);
        int w_id = (tid / num_anchors) % in_w;
        int anchor_sizes_id = (tid % anchor_sizes_size);
        int aspect_id = (tid / anchor_sizes_size) % aspect_ratios_size;
        Dtype x_ctr = w_id * stride_w + offset * (stride_w - 1);
        Dtype y_ctr = h_id * stride_h + offset * (stride_h - 1);
        float anchor_size = anchor_sizes_data[anchor_sizes_id];
        float ar = aspect_ratios_data[aspect_id];
        Dtype area = stride_w * stride_h;
        Dtype area_ratios = area / ar;
        Dtype base_w = round(sqrt(area_ratios));
        Dtype base_h = round(base_w * ar);
        Dtype scale_w = anchor_size / stride_w;
        Dtype scale_h = anchor_size / stride_h;
        Dtype half_width = 0.5 * (scale_w * base_w - 1);
        Dtype half_height = 0.5 * (scale_h * base_h - 1);
        Dtype* out_tmp = out_data + tid * 4;
        Dtype* var_tmp = var_data + tid * 4;
        out_tmp[0] = x_ctr - half_width;
        out_tmp[1] = y_ctr - half_height;
        out_tmp[2] = x_ctr + half_width;
        out_tmp[3] = y_ctr + half_height;
        var_tmp[0] = var_0;
        var_tmp[1] = var_1;
        var_tmp[2] = var_2;
        var_tmp[3] = var_3;
    }
}

template <DataType OpDtype>
SaberStatus SaberAnchorGenerator<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    AnchorGeneratorParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* var_data = (OpDataType*)outputs[1]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const float* anchor_sizes_data = (const float*)_anchor_sizes.data();
    const float* aspect_ratios_data = (const float*)_aspect_ratios.data();


    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int num_anchors = param.aspect_ratios.size() * param.anchor_sizes.size();
    int stride_h = param.stride[1];
    int stride_w = param.stride[0];
    float offset = param.offset;
    int count = in_h * in_w * num_anchors;

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_anchor_generator_fwd<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data, var_data, in_data, \
                 in_h, in_w, \
                 anchor_sizes_data,
                 param.anchor_sizes.size(), \
                 aspect_ratios_data,
                 param.aspect_ratios.size(), 
                 num_anchors,
                 stride_h, stride_w,
                 param.variances[0],
                 param.variances[1],
                 param.variances[2],
                 param.variances[3],
                 offset,
                 count);
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberAnchorGenerator, AnchorGeneratorParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAnchorGenerator, AnchorGeneratorParam, NV, AK_INT8);
}
}
