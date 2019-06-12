/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
#pragma OPENCL EXTENSION cl_amd_printf : enable
#if WIDTH == 1
typedef float _FLOAT;
#elif WIDTH == 2
typedef float2 _FLOAT;
#elif WIDTH == 4
typedef float4 _FLOAT;
#elif WIDTH == 8
typedef float8 _FLOAT;
#endif

#define CEILING(dividen, divisor) ((dividen + divisor - 1) / divisor)

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Relu(const __global _FLOAT* __restrict in,
     __global _FLOAT* __restrict out,
     int in_n,
     int in_c,
     int in_h,
     int in_w,
     int in_n_stride,
     int in_c_stride,
     int in_h_stride,
     int in_w_stride,
     int out_n_stride,
     int out_c_stride,
     int out_h_stride,
     int out_w_stride,
     float neg_slope) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];
        out[out_idx]  = (in_var > 0) ? in_var : in_var * neg_slope;
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Sigmoid(const __global _FLOAT* __restrict in,
        __global _FLOAT* __restrict out,
        int in_n,
        int in_c,
        int in_h,
        int in_w,
        int in_n_stride,
        int in_c_stride,
        int in_h_stride,
        int in_w_stride,
        int out_n_stride,
        int out_c_stride,
        int out_h_stride,
        int out_w_stride) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;
        _FLOAT in_var = in[in_idx];
        out[out_idx]  = (_FLOAT)(1 / (1 + exp(-in_var)));
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Tanh(const __global _FLOAT* __restrict in,
     __global _FLOAT* __restrict out,
     int in_n,
     int in_c,
     int in_h,
     int in_w,
     int in_n_stride,
     int in_c_stride,
     int in_h_stride,
     int in_w_stride,
     int out_n_stride,
     int out_c_stride,
     int out_h_stride,
     int out_w_stride) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];
        out[out_idx] = (float)(1) - ((float)(2) / ((float)(1) + exp(in_var * 2)));
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Stanh(const __global _FLOAT* __restrict in,
      __global _FLOAT* __restrict out,
      int in_n,
      int in_c,
      int in_h,
      int in_w,
      int in_n_stride,
      int in_c_stride,
      int in_h_stride,
      int in_w_stride,
      int out_n_stride,
      int out_c_stride,
      int out_h_stride,
      int out_w_stride,
      float slope,
      float coef) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];
        _FLOAT var    = in_var * slope;
        // output_data[j] = param.coef * tanh(param.negative_slope * in[j]);
        out[out_idx] =
            (_FLOAT)(coef * ((float)(1) - ((float)(2) / ((float)(1) + exp(var * 2)))));
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void Clipped_Relu(
    const __global _FLOAT* __restrict in,
    __global _FLOAT* __restrict out,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    int out_n_stride,
    int out_c_stride,
    int out_h_stride,
    int out_w_stride,
    float clipped_threadhold) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];
        in_var        = in_var > 0 ? in_var : 0;
        out[out_idx]  = (in_var < clipped_threadhold) ? in_var : clipped_threadhold;
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Elu(const __global _FLOAT* __restrict in,
    __global _FLOAT* __restrict out,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    int out_n_stride,
    int out_c_stride,
    int out_h_stride,
    int out_w_stride,
    float coef) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];
        out[out_idx]  = in_var > 0 ? in_var : coef * (exp(in_var) - 1);
    }
}

__attribute__((reqd_work_group_size(512, 1, 1)))
__kernel void
Prelu(const __global _FLOAT* __restrict in,
      __global _FLOAT* __restrict out,
      int in_n,
      int in_c,
      int in_h,
      int in_w,
      int in_n_stride,
      int in_c_stride,
      int in_h_stride,
      int in_w_stride,
      int out_n_stride,
      int out_c_stride,
      int out_h_stride,
      int out_w_stride,
      const __global float* __restrict slope,
      int is_channel_shared) {
    int count = CEILING(in_n * in_c * in_h * in_w, WIDTH);

    int global_size = get_global_size(0);
    int tid         = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int w = tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;

        int out_idx = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        _FLOAT in_var = in[in_idx];

        if (is_channel_shared) {
            out[out_idx] = (in_var > 0) ? in_var : (_FLOAT)(slope[0]) * in_var;
        } else {
            union {
                float slope[WIDTH];
                _FLOAT vec_slope;
            } transform;

            for (int i = 0; i < WIDTH; ++i) {
                int c_index = ((tid * WIDTH + i) / (in_h * in_w)) % in_c;
                transform.slope[i] = slope[c_index];
            }

            out[out_idx] = (in_var > 0) ? in_var : transform.vec_slope * in_var;
        }
    }
}
