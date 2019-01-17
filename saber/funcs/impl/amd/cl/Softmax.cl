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
#define SHARE_MEMORY_DIM 16384

//! general kernel for softmax
__kernel void softmax_max_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        _FLOAT min_data,
        int inner_num,
        int outer_num,
        int axis_size) {

    //! compute data index
    int idx = get_global_id(0);
    if (idx < total_size) {
        int idx_inner  = idx % inner_num;
        int idx_outer  = (idx / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;
        //! get maximum data across softmax axis
        _FLOAT max_data = min_data;
        for (int i = 0; i < axis_size; ++i) {
            max_data = in_data[real_index] > max_data ? in_data[real_index] : max_data;
            real_index += inner_num;
        }
        out_data[idx] = max_data;
    }
}

__kernel void softmax_max_roi_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        _FLOAT min_data,
        __global const int* input_stride_real,
        __global const int* output_stride_real,
        __global const int* shape_valid,
        int softmax_axis,
        int axis_size,
        int dims) {

    int idx = get_global_id(0);
    if (idx < total_size) {

        //! compute real data index
        int input_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                input_real_index += x * input_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        //! get maximum data across softmax axis
        _FLOAT max_data = min_data;
        for (int i = 0; i < axis_size; ++i) {
            max_data = in_data[input_real_index] > max_data ? in_data[input_real_index] : max_data;
            input_real_index += i * input_stride_real[softmax_axis];
        }
        out_data[idx] = max_data;
    }
}

__kernel void softmax_sub_exp_sum_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        __global const _FLOAT* max_data,
        __global _FLOAT* sum_data,
        int inner_num,
        int outer_num,
        int axis_size) {

    //! compute data index
    int idx = get_global_id(0);

    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;

        _FLOAT max_data_cur = max_data[idx];
        _FLOAT sum_data_cur = 0;
        int real_index      = idx_outer * inner_num + idx_inner;
        //! compute exp and summarize across the softmax axis
        for (int i = 0; i < axis_size; ++i) {
            _FLOAT sub_data = in_data[real_index] - max_data_cur;
            sub_data        = exp(sub_data);
            sum_data_cur += sub_data;
            out_data[real_index] = sub_data;
            real_index += inner_num;
        }
        sum_data[idx] = sum_data_cur;
    }
}

__kernel void softmax_sub_exp_sum_roi_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        __global const _FLOAT* max_data,
        __global _FLOAT* sum_data,
        __global const int* input_stride_real,
        __global const int* output_stride_real,
        __global const int* shape_valid,
        int softmax_axis,
        int axis_size,
        int dims) {

    //! compute data index
    int idx = get_global_id(0);

    if (idx < total_size) {
        //! compute real data index
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        _FLOAT max_data_cur = max_data[idx];
        _FLOAT sum_data_cur = 0;
        //! compute exp and summarize across the softmax axis
        for (int i = 0; i < axis_size; ++i) {
            _FLOAT sub_data = in_data[output_real_index] - max_data_cur;
            sub_data        = exp(sub_data);
            sum_data_cur += sub_data;
            out_data[output_real_index] = sub_data;
            output_real_index += output_stride_real[softmax_axis];
        }
        sum_data[idx] = sum_data_cur;
    }
}

__kernel void softmax_divid_output_kernel(
        int total_size,
        __global _FLOAT* io_data,
        __global const _FLOAT* sum_data,
        int inner_num,
        int outer_num,
        int axis_size) {
    //! compute data index
    int idx = get_global_id(0);

    if (idx < total_size) {
        int idx_inner       = idx % inner_num;
        int idx_outer       = (idx / inner_num) * axis_size;
        _FLOAT sum_data_cur = sum_data[idx];
        int real_index      = idx_outer * inner_num + idx_inner;
        //! compute final result
        for (int i = 0; i < axis_size; ++i) {
            io_data[real_index] = io_data[real_index] / sum_data_cur;
            real_index += inner_num;
        }
    }
}

__kernel void softmax_divid_output_roi_kernel(
        int total_size,
        __global _FLOAT* io_data,
        __global const _FLOAT* sum_data,
        __global const int* input_stride_real,
        __global const int* output_stride_real,
        __global const int* shape_valid,
        int softmax_axis,
        int axis_size,
        int dims) {
    //! compute data index
    int idx = get_global_id(0);

    if (idx < total_size) {
        //! compute real data index
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        _FLOAT sum_data_cur = sum_data[idx];
        //! compute final result
        for (int i = 0; i < axis_size; ++i) {
            io_data[output_real_index] = io_data[output_real_index] / sum_data_cur;
            output_real_index += output_stride_real[softmax_axis];
        }
    }
}

__kernel void sharemem_softmax_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        int inner_num,
        int outer_num,
        int axis_size) {

    __local _FLOAT data[SHARE_MEMORY_DIM];
    int tid = get_local_id(0);

    //! compute thread index and real data index
    int idx = get_global_id(0);

    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;
        // int blocksize = blockDim.x;
        int blocksize = get_local_size(0);

        int real_index = idx_outer * inner_num + idx_inner;
        int loop_idx   = real_index;
//! read all data to sharemem in softmax channel
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            data[tid + i * blocksize] = in_data[loop_idx];
            loop_idx += inner_num;
        }

        //! get maximum value in softmax channel
        _FLOAT max_data = data[tid];
#pragma unroll
        for (int i = 1; i < axis_size; ++i) {
            _FLOAT dt = data[tid + i * blocksize];
            if (max_data < dt) {
                max_data = dt;
            }
        }

        //! subtract then summarize
        _FLOAT sum = 0;
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            __local _FLOAT* dt = data + tid + i * blocksize;
            *dt                = exp(*dt - max_data);
            sum += *dt;
        }

        //! write back result
        loop_idx = real_index;
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            out_data[loop_idx] = data[tid + i * blocksize] / sum;
            loop_idx += inner_num;
        }
    }
}

__kernel void sharemem_softmax_roi_kernel(
        int total_size,
        __global const _FLOAT* in_data,
        __global _FLOAT* out_data,
        __global const int* input_stride_real,
        __global const int* output_stride_real,
        __global const int* shape_valid,
        int softmax_axis,
        int axis_size,
        int dims) {

    __local _FLOAT data[SHARE_MEMORY_DIM];
    int tid = get_local_id(0);

    //! compute thread index and real data index
    int idx1 = get_global_id(0);
    int idx  = idx1;

    if (idx < total_size) {

        int blocksize = get_local_size(0);

        //! compute real data index
        int input_real_index  = 0;
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                input_real_index += x * input_stride_real[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

//! read all data to sharemem in softmax channel
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            data[tid + i * blocksize] = in_data[input_real_index];
            input_real_index += input_stride_real[softmax_axis];
        }

        //! get maximum value in softmax channel
        _FLOAT max_data = data[tid];
#pragma unroll
        for (int i = 1; i < axis_size; ++i) {
            _FLOAT dt = data[tid + i * blocksize];
            if (max_data < dt) {
                max_data = dt;
            }
        }

        //! subtract then summarize
        _FLOAT sum = 0;
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            __local _FLOAT* dt = data + tid + i * blocksize;
            *dt                = exp(*dt - max_data);
            sum += *dt;
        }

//! write back result
#pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            out_data[output_real_index] = data[tid + i * blocksize] / sum;
            output_real_index += output_stride_real[softmax_axis];
        }
    }
}
