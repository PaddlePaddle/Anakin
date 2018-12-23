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
__kernel void ker_elt_production(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {
    int idx   = get_global_id(0);
    float tmp = in_data_a[idx] * in_data_b[idx];

    out_data[idx] = 1 == with_relu ? (tmp > 0 ? tmp : 0.0f) : tmp;

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
    float tmp = coeff1 * in_data1[idx] + coeff2 * in_data2[idx];

    out_data[idx] = 1 == with_relu ? (tmp > 0 ? tmp : 0.0f) : tmp;
}

__kernel void ker_elt_max(
    global float* out_data,
    const int with_relu,
    global const float* in_data_a,
    global const float* in_data_b,
    int count) {

    int idx = get_global_id(0);
    float tmp;
    float var_a = in_data_a[idx];
    float var_b = in_data_b[idx];
    int a_gt_b  = var_a > var_b;
    tmp         = a_gt_b ? var_a : var_b;

    out_data[idx] = 1 == with_relu ? (tmp > 0 ? tmp : 0.0f) : tmp;
}
