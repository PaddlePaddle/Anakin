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
#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

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
     _FLOAT neg_slope) {
    int count = in_n * in_c * in_h * in_w;

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
    int count = in_n * in_c * in_h * in_w;

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
    int count = in_n * in_c * in_h * in_w;

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
        // out[out_idx]  = (_FLOAT)((exp(in_var) - exp(-in_var)) / (exp(in_var) + exp(-in_var)));
        out[out_idx] = (_FLOAT)(1) - ((_FLOAT)(2) / ((_FLOAT)(1) + exp(in_var * 2)));
    }
}

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
      _FLOAT slope,
      _FLOAT coef) {
    int count = in_n * in_c * in_h * in_w;

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
                (_FLOAT)(coef * ((_FLOAT)(1) - ((_FLOAT)(2) / ((_FLOAT)(1) + exp(var * 2)))));
    }
}

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
        _FLOAT clipped_threadhold) {
    int count = in_n * in_c * in_h * in_w;

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
    _FLOAT coef) {
    int count = in_n * in_c * in_h * in_w;

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
      const __global _FLOAT* __restrict slope,
      int is_channel_shared) {
    int count = in_n * in_c * in_h * in_w;

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
            out[out_idx] = (in_var > 0) ? in_var : slope[0] * in_var;
        } else {
            out[out_idx] = (in_var > 0) ? in_var : slope[c] * in_var;
        }
    }
}
