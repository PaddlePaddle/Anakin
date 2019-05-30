

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

void adjust_small_heap_with_index_device(
    __local float* tree,
    __local float* index_tree,
    int index,
    int length) {
    while (2 * index + 1 < length) {
        int child_index = 2 * index + 1;

        if (child_index + 1 < length && tree[child_index + 1] < tree[child_index]) {
            child_index++;
        }

        if (tree[index] > tree[child_index]) {
            float t                 = tree[index];
            tree[index]             = tree[child_index];
            tree[child_index]       = t;
            int t_index             = index_tree[index];
            index_tree[index]       = index_tree[child_index];
            index_tree[child_index] = t_index;
            index                   = child_index;
        } else {
            break;
        }
    }
}
void adjust_small_heap_with_index_device_stride(
    __local float* tree,
    __local float* index_tree,
    int index,
    int length,
    int stride) {
    while (2 * index + 1 < length) {
        int child_index = 2 * index + 1;
        int off_0       = child_index * stride;
        int off_1       = (child_index + 1) * stride;

        if (child_index + 1 < length && tree[off_1] < tree[off_0]) {
            child_index++;
        }

        int child_off = child_index * stride;
        int cur_off   = index * stride;

        if (tree[cur_off] > tree[child_off]) {
            float t               = tree[cur_off];
            tree[cur_off]         = tree[child_off];
            tree[child_off]       = t;
            int t_index           = index_tree[cur_off];
            index_tree[cur_off]   = index_tree[child_off];
            index_tree[child_off] = t_index;
            index                 = child_index;
        } else {
            break;
        }
    }
}
__kernel void top1(__global const float* __restrict in_data,
     int height,
     int width,
     int out_max_val,
     __global float* __restrict out_data) {
    int group_id = get_group_id(0);

    if (group_id > height) {
        return;
    }

    __local float share_data[LOCAL_WORK_SIZE];
    __local float share_index[LOCAL_WORK_SIZE];
    int offset   = group_id * width;
    in_data      = in_data + offset;
    float minest = -1e32;
    int index    = get_local_id(0);

    if (index < width) {
        float result = in_data[index];
        int idx      = index;

        for (int tid = index + LOCAL_WORK_SIZE; tid < width; tid += LOCAL_WORK_SIZE) {
            if (result < in_data[tid]) {
                result = in_data[tid];
                idx    = tid;
            }
        }

        share_data[index]  = result;
        share_index[index] = idx;
    } else {
        share_data[index]  = minest;
        share_index[index] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (LOCAL_WORK_SIZE >= 512) {
        if (index < 256) {
            int index2 = index + 256;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (LOCAL_WORK_SIZE >= 256) {
        if (index < 128) {
            int index2 = index + 128;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (LOCAL_WORK_SIZE >= 128) {
        if (index < 64) {
            int index2 = index + 64;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index < 32) {
        if (LOCAL_WORK_SIZE >= 64) {
            int index2 = index + 32;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 32) {
            int index2 = index + 16;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 16) {
            int index2 = index + 8;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 8) {
            int index2 = index + 4;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 4) {
            int index2 = index + 2;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 2) {
            int index2 = index + 1;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (index == 0) {
        if (!out_max_val) {
            out_data[group_id] = share_index[0];
        } else {
            out_data[2 * group_id]     = share_index[0];
            out_data[2 * group_id + 1] = share_data[0];
        }
    }
}

__kernel void block_top1(
    __global const float* __restrict in_data,
    int height,
    int width,
    int inner_group_num,
    __global float* __restrict out_data,
    __global float* __restrict out_index) {
    __local float share_data[LOCAL_WORK_SIZE];
    __local float share_index[LOCAL_WORK_SIZE];
    int group_id = get_group_id(0);
    int batch_id = group_id / inner_group_num;
    int inner_group_id = group_id - batch_id * inner_group_num;
    int offset   = batch_id * width + inner_group_id * LOCAL_WORK_SIZE;
    in_data      = in_data + offset;
    float minest = -1e32;
    int index    = get_local_id(0);

    if (index + inner_group_id * LOCAL_WORK_SIZE < width) {
        share_data[index]  = in_data[index];
        share_index[index] = index;
    } else {
        share_data[index]  = minest;
        share_index[index] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (LOCAL_WORK_SIZE >= 256) {
        if (index < 128) {
            int index2 = index + 128;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (LOCAL_WORK_SIZE >= 128) {
        if (index < 64) {
            int index2 = index + 64;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index < 32) {
        if (LOCAL_WORK_SIZE >= 64) {
            int index2 = index + 32;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 32) {
            int index2 = index + 16;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 16) {
            int index2 = index + 8;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 8) {
            int index2 = index + 4;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 4) {
            int index2 = index + 2;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 2) {
            int index2 = index + 1;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (index == 0) {
        int offset        = group_id;
        out_data[offset]  = share_data[0];
        out_index[offset] = share_index[0];
    }
}

__kernel void top1_big(
    __global const float* __restrict in_data,
    __global const float* __restrict in_index,
    int height,
    int width,
    int out_max_val,
    __global float* __restrict out_data) {
    __local float share_data[LOCAL_WORK_SIZE];
    __local float share_index[LOCAL_WORK_SIZE];
    int group_id = get_group_id(0);
    int offset   = group_id * width;
    in_data      = in_data + offset;
    in_index     = in_index + offset;
    float minest = -1e10;
    int index    = get_local_id(0);

    if (index < width) {
        float result = in_data[index];
        int idx      = index;

        for (int tid = index + LOCAL_WORK_SIZE; tid < width; tid += LOCAL_WORK_SIZE) {
            if (result < in_data[tid]) {
                result = in_data[tid];
                idx    = tid;
            }
        }

        share_data[index]  = result;
        share_index[index] = idx;
    } else {
        share_data[index]  = minest;
        share_index[index] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (LOCAL_WORK_SIZE >= 256) {
        if (index < 128) {
            int index2 = index + 128;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (LOCAL_WORK_SIZE >= 128) {
        if (index < 64) {
            int index2 = index + 64;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index < 32) {
        if (LOCAL_WORK_SIZE >= 64) {
            int index2 = index + 32;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 32) {
            int index2 = index + 16;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 16) {
            int index2 = index + 8;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 8) {
            int index2 = index + 4;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 4) {
            int index2 = index + 2;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (LOCAL_WORK_SIZE >= 2) {
            int index2 = index + 1;

            if (share_data[index2] > share_data[index]) {
                share_data[index]  = share_data[index2];
                share_index[index] = share_index[index2];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (index == 0) {
        int block_id = share_index[0];

        if (!out_max_val) {
            out_data[group_id] = block_id * LOCAL_WORK_SIZE + in_index[block_id];
        } else {
            out_data[2 * group_id]     = block_id * LOCAL_WORK_SIZE + in_index[block_id];
            out_data[2 * group_id + 1] = share_data[0];
        }
    }
}

__kernel void top1_channel(
    __global const float* __restrict in_data,
    int num,
    int channel,
    int inner_dim,
    int out_max_val,
    __global float* __restrict out_data) {
    int local_id  = get_local_id(0);
    int group_id  = get_group_id(0);
    int thread_id = local_id + group_id * LOCAL_WORK_SIZE;

    if (thread_id >= num * inner_dim) {
        return;
    }

    int num_id   = thread_id / inner_dim;
    int inner_id = thread_id % inner_dim;

    in_data        = in_data + num_id * channel * inner_dim + inner_id;
    float max_data = in_data[0];
    int max_id     = 0;

    for (int i = 1; i < channel; i++) {
        float data = in_data[i * inner_dim];

        if (max_data < data) {
            max_data = data;
            max_id   = i;
        }
    }

    out_data[thread_id] = out_max_val ? max_data : max_id;
}

__kernel void topk_channel(
    __global const float* __restrict in_data,
    int num,
    int channel,
    int inner_dim,
    int top_k,
    int out_max_val,
    __global float* __restrict out_data) {
    int zero       = 0;
    int local_id   = get_local_id(0);
    int group_id   = get_group_id(0);
    int local_size = get_local_size(0);
    int thread_id  = local_id + group_id * local_size;

    if (thread_id >= num * inner_dim) {
        return;
    }

    int num_id   = thread_id / inner_dim;
    int inner_id = thread_id % inner_dim;
    //
    in_data = in_data + num_id * channel * inner_dim + inner_id;
    __local float trees[TREE_MEM_SIZE];
    __local float* small_heap_tree = trees + local_id * top_k;
    __local float* tree_index      = trees + local_id * top_k + local_size * top_k;

    for (int i = 0; i < top_k; i++) {
        small_heap_tree[i] = -FLT_MAX;
        tree_index[i]      = -1;
    }

    for (int i = 0; i < channel; i++) {
        float data = in_data[i * inner_dim];

        if (data > small_heap_tree[0]) {
            small_heap_tree[0] = data;
            tree_index[0]      = i;
            adjust_small_heap_with_index_device(small_heap_tree, tree_index, zero, top_k);
        }
    }

    out_data = out_data + num_id * top_k * inner_dim + inner_id;

    for (int i = top_k - 1; i >= 0; i--) {
        out_data[i * inner_dim] = out_max_val ? small_heap_tree[0] : tree_index[0];
        small_heap_tree[0]      = FLT_MAX;
        tree_index[0]           = -1;
        adjust_small_heap_with_index_device(small_heap_tree, tree_index, zero, top_k);
    }
}
__kernel void topk_heap_shared(
    __global float* __restrict out_data,
    int n,
    int inner_dim,
    int top_k,
    int out_max_val,
    __global const float* __restrict in_data) {

    int zero = 0;
    __local float trees[TREE_MEM_SIZE];
    const int group_id            = get_group_id(0);
    const int tid                 = get_local_id(0);
    const int local_size          = get_local_size(0);
    __local float* cur_tree       = trees + tid * top_k;
    __local float* cur_tree_index = cur_tree + top_k * local_size;

    for (int i = 0; i < top_k; i++) {
        cur_tree[i]       = -FLT_MAX;
        cur_tree_index[i] = -1;
    }

    /*build small heap for every thread in one picture*/
    in_data = in_data + group_id * inner_dim;
    int ttt = 0;

    for (int i = tid; i < inner_dim; i += local_size) {
        if (in_data[i] > cur_tree[0]) {
            cur_tree[0]       = in_data[i];
            cur_tree_index[0] = i;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= 256) {
        if (tid < 128) {
            __local float* next_tree       = cur_tree + 128 * top_k;
            __local float* next_tree_index = cur_tree_index + 128 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 128) {
        if (tid < 64) {
            __local float* next_tree       = cur_tree + 64 * top_k;
            __local float* next_tree_index = cur_tree_index + 64 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 64) {
        if (tid < 32) {
            __local float* next_tree       = cur_tree + 32 * top_k;
            __local float* next_tree_index = cur_tree_index + 32 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 32) {
        if (tid < 16) {
            __local float* next_tree       = cur_tree + 16 * top_k;
            __local float* next_tree_index = cur_tree_index + 16 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 16) {
        if (tid < 8) {
            __local float* next_tree       = cur_tree + 8 * top_k;
            __local float* next_tree_index = cur_tree_index + 8 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 8) {
        if (tid < 4) {
            __local float* next_tree       = cur_tree + 4 * top_k;
            __local float* next_tree_index = cur_tree_index + 4 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 4) {
        if (tid < 2) {
            __local float* next_tree       = cur_tree + 2 * top_k;
            __local float* next_tree_index = cur_tree_index + 2 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 2) {
        if (tid < 1) {
            __local float* next_tree       = cur_tree + 1 * top_k;
            __local float* next_tree_index = cur_tree_index + 1 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        int stride = out_max_val ? group_id * top_k * 2 : group_id * top_k;
        out_data   = out_data + stride;

        for (int i = top_k - 1; i >= 0; i--) {
            if (!out_max_val) {
                out_data[i] = cur_tree_index[0];
            } else {
                out_data[i + top_k]         = cur_tree[0];
                out_data[i] = cur_tree_index[0];
            }

            cur_tree[0]       = FLT_MAX;
            cur_tree_index[0] = -1;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
        }
    }
}
__attribute__((reqd_work_group_size(512,1,1)))
__kernel void topk_heap_shared_512(
    __global float* __restrict out_data,
    int n,
    int inner_dim,
    int top_k,
    int out_max_val,
    __global const float* __restrict in_data) {

    int zero = 0;
    __local float trees[TREE_MEM_SIZE];
    const int group_id            = get_group_id(0);
    const int tid                 = get_local_id(0);
    const int local_size          = get_local_size(0);
    __local float* cur_tree       = trees + tid * top_k;
    __local float* cur_tree_index = cur_tree + top_k * local_size;

    for (int i = 0; i < top_k; i++) {
        cur_tree[i]       = -FLT_MAX;
        cur_tree_index[i] = -1;
    }

    /*build small heap for every thread in one picture*/
    in_data = in_data + group_id * inner_dim;
    int ttt = 0;

    for (int i = tid; i < inner_dim; i += local_size) {
        if (in_data[i] > cur_tree[0]) {
            cur_tree[0]       = in_data[i];
            cur_tree_index[0] = i;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_size >= 512) {
        if (tid < 256) {
            __local float* next_tree       = cur_tree + 256 * top_k;
            __local float* next_tree_index = cur_tree_index + 256 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 256) {
        if (tid < 128) {
            __local float* next_tree       = cur_tree + 128 * top_k;
            __local float* next_tree_index = cur_tree_index + 128 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 128) {
        if (tid < 64) {
            __local float* next_tree       = cur_tree + 64 * top_k;
            __local float* next_tree_index = cur_tree_index + 64 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 64) {
        if (tid < 32) {
            __local float* next_tree       = cur_tree + 32 * top_k;
            __local float* next_tree_index = cur_tree_index + 32 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 32) {
        if (tid < 16) {
            __local float* next_tree       = cur_tree + 16 * top_k;
            __local float* next_tree_index = cur_tree_index + 16 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 16) {
        if (tid < 8) {
            __local float* next_tree       = cur_tree + 8 * top_k;
            __local float* next_tree_index = cur_tree_index + 8 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 8) {
        if (tid < 4) {
            __local float* next_tree       = cur_tree + 4 * top_k;
            __local float* next_tree_index = cur_tree_index + 4 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 4) {
        if (tid < 2) {
            __local float* next_tree       = cur_tree + 2 * top_k;
            __local float* next_tree_index = cur_tree_index + 2 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 2) {
        if (tid < 1) {
            __local float* next_tree       = cur_tree + 1 * top_k;
            __local float* next_tree_index = cur_tree_index + 1 * top_k;

            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0]       = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        int stride = out_max_val ? group_id * top_k * 2 : group_id * top_k;
        out_data   = out_data + stride;

        for (int i = top_k - 1; i >= 0; i--) {
            if (!out_max_val) {
                out_data[i] = cur_tree_index[0];
            } else {
                out_data[i + top_k]         = cur_tree[0];
                out_data[i] = cur_tree_index[0];
            }

            cur_tree[0]       = FLT_MAX;
            cur_tree_index[0] = -1;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, zero, top_k);
        }
    }
}
