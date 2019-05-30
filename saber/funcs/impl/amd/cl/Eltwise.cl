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
__kernel void ker_elt_production(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {
    int idx   = get_global_id(0);

    if (idx < count) {
        float tmp = in_data_a[idx] * in_data_b[idx];
#if MLO_CONV_ACTIVE_RELU
        out_data[idx] = tmp > 0 ? tmp : 0.0f;
#else
        out_data[idx] = tmp;
#endif
    }

}

__kernel void ker_elt_production_f4(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {
    int idx   = get_global_id(0);
    float4 tmp1, tmp2, tmp3;
    global float4*   vdst;

    if (idx < count) {
        tmp1 = *((__global const float4*)(&in_data_a[idx * 4]));
        tmp2 = *((__global const float4*)(&in_data_b[idx * 4]));
        tmp3.x = tmp1.x * tmp2.x;
        tmp3.y = tmp1.y * tmp2.y;
        tmp3.z = tmp1.z * tmp2.z;
        tmp3.w = tmp1.w * tmp2.w;

#if MLO_CONV_ACTIVE_RELU
        tmp3.x = tmp3.x > 0 ? tmp3.x : 0.0f;
        tmp3.y = tmp3.y > 0 ? tmp3.y : 0.0f;
        tmp3.z = tmp3.z > 0 ? tmp3.z : 0.0f;
        tmp3.w = tmp3.w > 0 ? tmp3.w : 0.0f;
#endif
        vdst = (__global float4*)(&out_data[idx * 4]);
        *vdst = tmp3;
    }

}

__kernel void ker_elt_sum(
    global float* out_data,
    const int with_relu,
    global const float* in_data1,
    global const float* in_data2,
    float coeff1,
    float coeff2,
    int count) {
    int idx   = get_global_id(0);

    if (idx < count) {
        float tmp = coeff1 * in_data1[idx] + coeff2 * in_data2[idx];
#if MLO_CONV_ACTIVE_RELU
        out_data[idx] = tmp > 0 ? tmp : 0.0f;
#else
        out_data[idx] = tmp;
#endif
    }
}

__kernel void ker_elt_sum_f4(
    global float* out_data,
    const int with_relu,
    global const float* in_data1,
    global const float* in_data2,
    float coeff1,
    float coeff2,
    int count) {

    float4 tmp1, tmp2, tmp3;
    global float4*   vdst;

    int idx   = get_global_id(0);

    if (idx < count) {
        tmp1 = *((__global const float4*)(&in_data1[idx * 4]));
        tmp2 = *((__global const float4*)(&in_data2[idx * 4]));
        tmp3.x = coeff1 * tmp1.x + coeff2 * tmp2.x;
        tmp3.y = coeff1 * tmp1.y + coeff2 * tmp2.y;
        tmp3.z = coeff1 * tmp1.z + coeff2 * tmp2.z;
        tmp3.w = coeff1 * tmp1.w + coeff2 * tmp2.w;
#if MLO_CONV_ACTIVE_RELU
        tmp3.x = tmp3.x > 0 ? tmp3.x : 0.0f;
        tmp3.y = tmp3.y > 0 ? tmp3.y : 0.0f;
        tmp3.z = tmp3.z > 0 ? tmp3.z : 0.0f;
        tmp3.w = tmp3.w > 0 ? tmp3.w : 0.0f;
#endif
        vdst = (__global float4*)(&out_data[idx * 4]);
        *vdst = tmp3;
    }
}

__kernel void ker_elt_max(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {

    int idx = get_global_id(0);

    if (idx < count) {
        float tmp;
        float var_a = in_data_a[idx];
        float var_b = in_data_b[idx];
        int a_gt_b  = var_a > var_b;
        tmp         = a_gt_b ? var_a : var_b;

#if MLO_CONV_ACTIVE_RELU
        out_data[idx] = tmp > 0 ? tmp : 0.0f;
#else
        out_data[idx] = tmp;
#endif
    }
}

__kernel void ker_elt_max_f4(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {

    int idx = get_global_id(0);
    float4 tmp1, tmp2;
    global float4*   vdst;

    if (idx < count) {
        tmp1 = (__global const float4)(in_data_a[idx]);
        tmp2 = (__global const float4)(in_data_b[idx]);
        //int a_gt_b  = var_a > var_b;
        //tmp         = a_gt_b ? var_a : var_b;
        tmp1.x = tmp1.x > tmp2.x ? tmp1.x : tmp2.x;
        tmp1.y = tmp1.y > tmp2.y ? tmp1.y : tmp2.y;
        tmp1.z = tmp1.z > tmp2.z ? tmp1.z : tmp2.z;
        tmp1.w = tmp1.w > tmp2.w ? tmp1.w : tmp2.w;
#if MLO_CONV_ACTIVE_RELU
        tmp1.x = tmp1.x > 0 ? tmp1.x : 0.0f;
        tmp1.y = tmp1.y > 0 ? tmp1.y : 0.0f;
        tmp1.z = tmp1.z > 0 ? tmp1.z : 0.0f;
        tmp1.w = tmp1.w > 0 ? tmp1.w : 0.0f;
#endif
        vdst = (__global float4*)(&out_data[idx]);
        *vdst = tmp1;

    }
}
