#include "saber/funcs/impl/cuda/saber_deformable_conv.h"
#include "cuda_fp16.h"
#include "saber/core/tensor_op.h"

namespace anakin {
namespace saber {

__device__ float deformable_im2col_bilinear(const float* bottom_data, const int data_width,
                                            const int height, const int width, float h, float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;
    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (float) h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (float) w_low;
    } else {
        w_high = w_low + 1;
    }
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;
    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__global__ void deformable_im2col_gpu_kernel(const int n, const float* data_im,
                                             const float* data_offset, const int height, const int width,
                                             const int kernel_h, const int kernel_w, const int pad_h,
                                             const int pad_w, const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w,
                                             const int channel_per_deformable_group, const int height_col,
                                             const int width_col, float* data_col) {

    CUDA_KERNEL_LOOP(index, n) {
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col) / height_col;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        // THIS IS THE TRUE CHANNEL
        const int deformable_group_index = c_im / channel_per_deformable_group;

        //input map coord(h_in, w_in)
        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        //data_col (data & offset)
        float* data_col_ptr = data_col
                              + (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + (c_im * height + h_in) * width
                                   + w_in;
        const float* data_offset_ptr = data_offset
                                       + deformable_group_index * 2 * kernel_h * kernel_w * height_col
                                         * width_col;

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                //offset_h and offset_w in the same channel
                const int data_offset_h_ptr = ((2 * (i * kernel_w + j))
                                               * height_col + h_col) * width_col + w_col;

                const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1)
                                               * height_col + h_col) * width_col + w_col;
                const float offset_h = data_offset_ptr[data_offset_h_ptr];
                const float offset_w = data_offset_ptr[data_offset_w_ptr];
                float val = 0.f;
                const float h_im = h_in + i * dilation_h + offset_h;
                const float w_im = w_in + j * dilation_w + offset_w;
                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                    const float map_h = i * dilation_h + offset_h;
                    const float map_w = j * dilation_w + offset_w;
                    // cur_height (from h_in to height)
                    const int cur_height = height - h_in;
                    const int cur_width = width - w_in;
                    val = deformable_im2col_bilinear(data_im_ptr, width,
                                                     cur_height, cur_width, map_h, map_w);
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}
__global__ void gpu_add_bias(float * out_data, const int count,
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride,
                             int in_h_stride, int in_w_stride,
                             const float *bias) {
    CUDA_KERNEL_LOOP(tid, count){
        int read_w =  tid % in_w;
        int read_h = (tid / (in_w)) % in_h;
        int read_c = (tid / (in_h * in_w)) % in_c;
        int read_n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = read_n * in_n_stride
                     + read_c * in_c_stride
                     + read_h * in_h_stride
                     + read_w * in_w_stride;

        float in_var = out_data[in_idx];
        float in_bias = bias[read_c];
        out_data[in_idx] = in_var + in_bias;
    }
}
template <>
SaberStatus SaberDeformableConv2D<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            DeformableConvParam<NV>& param) {
                                
    int in_channel = inputs[0]->channel();
    int conv_out_channel = outputs[0]->channel();

    const OpDataType* weight = (const float*)param.weight()->data();
    const OpDataType* data = (const float*)inputs[0]->data();
    const OpDataType* offset = (const float*)inputs[1]->data();

    OpDataType* top_data = (float*)outputs[0]->mutable_data();

    OpDataType* deformable_col_buffer_data = (float*)_deform_col_buffer.mutable_data();
    const OpDataType* deform_col_buffer_data_const = (const float*)_deform_col_buffer.data();

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    for (int n = 0; n < inputs[0]->num(); ++n) {

        // transform image to col_buffer in order to use gemm

        int channel_per_group = in_channel / param.group;
        int num_kernels = in_channel * _deform_col_buffer.height() * _deform_col_buffer.width();

        deformable_im2col_gpu_kernel
                <<<CUDA_GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                        num_kernels, data + n * _bottom_dim, offset + n * _offset_dim,
                        inputs[0]->height(), inputs[0]->width(),
                        param.weight()->height(), param.weight()->width(),
                        param.pad_h, param.pad_w, param.stride_h, param.stride_w,
                        param.dilation_h, param.dilation_w,
                        channel_per_group, _deform_col_buffer.height(),
                        _deform_col_buffer.width(),
                        deformable_col_buffer_data);
        
        OpDataType* out_data = top_data + n * _output_offset;

        for (int g = 0; g < param.group; ++g) {
            float alpha = 1.f;
            float beta = 0.f;
            CUBLAS_CHECK(cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        _conv_out_spatial_dim,
                        conv_out_channel / param.group,
                        _kernel_dim / param.group,
                        &alpha,
                        deform_col_buffer_data_const + _col_offset / param.group * g,
                        _conv_out_spatial_dim,
                        weight + _kernel_offset / param.group * g,
                        _kernel_dim / param.group,
                        &beta,
                        out_data + _output_offset / param.group * g,
                        _conv_out_spatial_dim));
        }
        if (param.bias()->size() > 0) {
            Shape out_shape = outputs[0]->valid_shape();
            Shape out_stride = outputs[0]->get_stride();
            int out_count = outputs[0]->size();
            const float* bias_data = (const float*)param.bias()->data();
            gpu_add_bias<<<CUDA_GET_BLOCKS(out_count), CUDA_NUM_THREADS, 0, cuda_stream>>> (out_data, out_count,
                            out_shape[0], out_shape[1],
                            out_shape[2], out_shape[3],
                            out_stride[0], out_stride[1],
                            out_stride[2], out_stride[3],
                            bias_data);
        }
        CUDA_POST_KERNEL_CHECK;
    }

    return SaberSuccess;
}
template class SaberDeformableConv2D<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeformableConv2D, DeformableConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDeformableConv2D, DeformableConvParam, NV, AK_INT8);

}
}