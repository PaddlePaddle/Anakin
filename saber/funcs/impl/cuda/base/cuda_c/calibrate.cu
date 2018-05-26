#include "anakin_config.h"
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/core/common.h"
#include <stdio.h>

namespace anakin {
namespace saber{

    __global__ void ker_calibrate_from_fp32_to_int8(void* data_int8,
                        const void* data_fp32, int size, float scale,
                        int in_n, int in_c, int in_h, int in_w,
                        int stride_n, int stride_c, int stride_h, int stride_w) {

        CUDA_KERNEL_LOOP(tid, size){
            int read_w =  tid % in_w;
            int read_h = (tid / (in_w)) % in_h;
            int read_c = (tid / (in_h * in_w)) % in_c;
            int read_n = (tid / (in_c * in_h * in_w)) % in_n;

            int in_idx = read_n * stride_n
                         + read_c * stride_c
                         + read_h * stride_h
                         + read_w * stride_w;

            float* data_in = (float*)data_fp32;
            if (scale <= 1e-6) {
                scale = 1e-6;
            }

            ((char*)data_int8)[tid] = (char)(data_in[in_idx] / scale);
//            printf("%f, ", data_in[in_idx] /scale);
        }
    }

    __global__ void ker_calibrate_from_int8_to_fp32(void* data_fp32,
                        const void* data_int8, int size, float scale,
                        int out_n, int out_c, int out_h, int out_w,
                        int stride_n, int stride_c, int stride_h, int stride_w) {

        CUDA_KERNEL_LOOP(tid, size) {

            int write_w =  tid % out_w;
            int write_h = (tid / (out_w)) % out_h;
            int write_c = (tid / (out_h * out_w)) % out_c;
            int write_n = (tid / (out_c * out_h * out_w)) % out_n;

            int out_idx = write_n * stride_n
                          + write_c * stride_c
                          + write_h * stride_h
                          + write_w * stride_w;

            const char* data_in = (const char*)data_int8;
            if (scale <= 1e-6) {
                scale = 1e-6;
            }
            ((float*)data_fp32)[out_idx] = (float)(data_in[tid]) * scale;
//            printf("%d, ", data_in[tid]);
        }

    }

    void calibrate_to_int8(void* data_int8, const void* data_fp32, int size, float* scale, Context<NV> ctx,
                           int in_n, int in_c, int in_h, int in_w,
                           int stride_n, int stride_c, int stride_h, int stride_w) {

        cudaStream_t cuda_stream = ctx.get_compute_stream();
        ker_calibrate_from_fp32_to_int8<<<CUDA_GET_BLOCKS(size),
                CUDA_NUM_THREADS, 0, cuda_stream>>>(data_int8, data_fp32, size, *scale,
                in_n, in_c, in_h, in_w,
                stride_n, stride_c, stride_h, stride_w);

    }

    void calibrate_to_fp32(void* data_fp32, const void* data_int8, int size, float* scale, Context<NV> ctx,
                           int out_n, int out_c, int out_h, int out_w,
                           int stride_n, int stride_c, int stride_h, int stride_w) {

        cudaStream_t cuda_stream = ctx.get_compute_stream();
        ker_calibrate_from_int8_to_fp32<<<CUDA_GET_BLOCKS(size),
                CUDA_NUM_THREADS, 0, cuda_stream>>>(data_fp32, data_int8, size, *scale,
                out_n, out_c, out_h, out_w,
                stride_n, stride_c, stride_h, stride_w);

    }

}
}