
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/funcs/calibrate.h"
#include <cfloat>

namespace anakin {
namespace saber {

__global__
void transform_nchw_2_c4(char* out_data, const float* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         float scale,
                         int count) {

    int load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int write_w = (gid) % valid_width;
    int write_h = (gid / (out_h_stride)) % valid_height;
    int write_c = (gid / (out_c_stride)) % valid_channel_4;
    int write_n = (gid / (out_n_stride)) % valid_num;

    int in_offset = write_n * in_n_stride
                    + write_c * in_c_stride * 4
                    + write_h * in_h_stride
                    + write_w * in_w_stride;

    int out_offset = write_n * out_n_stride
                     + write_c * out_c_stride
                     + write_h * out_h_stride
                     + write_w;

    if (gid < count) {
        char4 write;
        load0 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.x = static_cast<char>(load0);

        in_offset += in_c_stride;
        load1 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.y = static_cast<char>(load1);

        in_offset += in_c_stride;
        load2 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.z = static_cast<char>(load2);

        in_offset += in_c_stride;
        load3 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.w = static_cast<char>(load3);

        ((char4*)out_data)[out_offset] = write;
    }
}

template<>
SaberStatus conv_calibrate_fp32_int8_c4<NV>(Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor, const float in_scale, Context<NV> ctx) {

    const float * in_data = (const float*)in_tensor.data();
    char * out_data = (char*)out_tensor.mutable_data();

    Shape in_stride = in_tensor.get_stride();

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    transform_nchw_2_c4<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
        0, cuda_stream>>>(out_data, in_data,
            out_shape[0], out_shape[1], out_shape[2], out_shape[3],
            in_stride[0], in_stride[1], in_stride[2], in_stride[3],
            out_shape[1] * out_shape[2] * out_shape[3],
            out_shape[2] * out_shape[3], out_shape[3], 1,
            (1.f / in_scale), count);

    return SaberSuccess;
}

__global__ void transform_nchw_2_nchw(float * out_data,
                                      const float* in_data, const int count,
                                      int in_n, int in_c, int in_h, int in_w,
                                      int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                                      int out_n, int out_c, int out_h, int out_w,
                                      int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                                      const float *scale, const float input_scale) {
    CUDA_KERNEL_LOOP(tid, count){
        int read_w =  tid % in_w;
        int read_h = (tid / (in_w)) % in_h;
        int read_c = (tid / (in_h * in_w)) % in_c;
        int read_n = (tid / (in_c * in_h * in_w)) % in_n;

        int write_w =  tid % out_w;
        int write_h = (tid / (out_w)) % out_h;
        int write_c = (tid / (out_h * out_w)) % out_c;
        int write_n = (tid / (out_c * out_h * out_w)) % out_n;

        int in_idx = read_n * in_n_stride
                     + read_c * in_c_stride
                     + read_h * in_h_stride
                     + read_w * in_w_stride;

        int out_idx = write_n * out_n_stride
                      + write_c * out_c_stride
                      + write_h * out_h_stride
                      + write_w * out_w_stride;

        float in_var = in_data[in_idx];
        float in_scale = scale[read_c];
        out_data[out_idx] = in_var * in_scale * input_scale;
    }
}

template<>
SaberStatus conv_calibrate_int32_fp32<NV>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor,
        const float in_scale, const float* weight_scale, Context<NV> ctx) {

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    Shape stride_in = in_tensor.get_stride();
    Shape stride_out = out_tensor.get_stride();

    const float *in_data = (const float*)in_tensor.data();
    float *out_data = (float*)out_tensor.mutable_data();

    const int count = in_tensor.valid_size();
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_nchw
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    weight_scale, in_scale);

    return SaberSuccess;
}

__global__
void int8nchwc4_fp32nchw(float* out_data, const char* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         const float* scale, int count) {

    float load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;
    int scale_index = read_c << 2;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {

        char4 readin = __ldg(&((const char4*)in_data)[in_offset]);

        load0 = static_cast<float>(readin.x);
        load1 = static_cast<float>(readin.y);
        load2 = static_cast<float>(readin.z);
        load3 = static_cast<float>(readin.w);

        out_data[out_offset] = load0 * scale[scale_index]; out_offset += out_c_stride;
        out_data[out_offset] = load1 * scale[scale_index + 1]; out_offset += out_c_stride;
        out_data[out_offset] = load2 * scale[scale_index + 2]; out_offset += out_c_stride;
        out_data[out_offset] = load3 * scale[scale_index + 3];
    }
}

template<>
SaberStatus conv_calibrate_int8_c4_fp32<NV>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        const float* weight_scale,
        Context<NV> ctx) {

    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];

    const char * in_data = (const char*)in_tensor.data();
    float * out_data = (float*)out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    int8nchwc4_fp32nchw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            in_shape[1] * in_shape[2] * in_shape[3],
            in_shape[2] * in_shape[3],
            in_shape[3], 1,
            out_stride[0], out_stride[1], out_stride[2], out_stride[3],
            weight_scale, count);

    return SaberSuccess;
}

#define JUDGESIGN(x) (((x) >= 0) ? +1 : -1)

__global__
void calibrate_float2char_col(signed char* dst, const float* src,
                              float * scale, int height, int width) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float col_max = 0.0f;
    const float *data = src + gid;
    for(int idx = 0; idx < height; ++idx){
        if (gid < width) {
            float temp = fabsf(data[idx * width]);
            col_max = (col_max >= temp)? col_max : temp;
        }
    }
    signed char* target = dst + gid;
    float col_scale = (float)((1 << 7) - 1) / col_max;
    for(int idx = 0; idx < height; ++idx) {
        if(gid < width) {
            float temp = data[idx * width];
            if(temp >= col_max - FLT_EPSILON) {
                target[idx * width] = (signed char)((1 << 7) - 1);
            } else if(temp <= -col_max + FLT_EPSILON) {
                target[idx * width] = (signed char)(-(1 << 7));
            } else {
                target[idx * width] = (signed char)(temp * col_scale + JUDGESIGN(temp) * 0.5);
            }
        }
    }
    scale[gid] = 1.f / col_scale;
}

__global__
void calibrate_float2char_row(signed char* dst, const float* src,
                                         float * scale, int height, int width) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float row_max = 0.0f;
    const float * data = src + width * gid;
    for(int idx = 0; idx < width; ++idx) {
        if(gid < height){
            float temp = fabsf(data[idx]);
            row_max = (row_max >= temp) ? row_max : temp;
        }
    }
    signed char * target = dst + width * gid;
    float row_scale = (float)((1 << 7) - 1) / row_max;
    for(int idx = 0; idx < width; ++idx) {
        if(gid < height) {
            float temp = data[idx];
            if(temp >= row_max - FLT_EPSILON) {
                target[idx] = (signed char)((1 << 7) - 1);
            } else if(temp <= -row_max + FLT_EPSILON) {
                target[idx] = (signed char)(-(1 << 7));
            } else {
                target[idx] = (signed char)(temp * row_scale + JUDGESIGN(temp) * 0.5);
            }
        }
    }
    scale[gid] = 1.f / row_scale;
}

__global__ void calibrate_fix2float(float * dst,
                                    const float* sA, const float* sB,
                                    float alpha, float beta, int height,
                                    int width, int threads) {
    int ri = blockIdx.x;
    int tid = threadIdx.x;
    int loop = (width / threads) + ((width % threads == 0) ? 0 : 1);

    float rscale = (sA[ri] == 0.0f) ? 1.0f : sA[ri];
    float * data = dst + width * ri;
    int idx = 0;
    for (int i = 0; i < loop; ++i) {
        if(idx + tid < width){
            float temp = data[idx + tid];
            float cscale = (sB[idx + tid] == 0.0f) ? 255.0f : sB[idx + tid];
            data[idx + tid] = beta  * temp + alpha * temp * rscale * cscale;
        }
        idx += threads;
    }
}

template <>
void float2char<NV>(bool col_direct, signed char* dst, const float* src,
                    float *scale, int height, int width, Context<NV> ctx) {
    int threads = 32;
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    if (col_direct) {
        calibrate_float2char_col <<< (width / threads) + (((width % threads) == 0) ? 0 : 1), threads, 0,
                cuda_stream >>> (
                        dst, src, scale, height, width);
    } else {
        calibrate_float2char_row<<<(height / threads) + (((height % threads)==0) ? 0 : 1), threads, 0, cuda_stream>>>(
                dst, src, scale, height, width);
    }
}
template <>
void fix2float<NV>(float * dst,
               const float *sA, const float *sB,
               const float alpha, const float beta, int height, int width, Context<NV> ctx) {
    int threads = 256;
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    calibrate_fix2float<<<height, threads, 0, cuda_stream>>>(dst, sA, sB, alpha, beta,
            height, width, threads);
}
}
}