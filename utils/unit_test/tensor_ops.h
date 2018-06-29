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

#ifndef ANAKIN2_UTILS_TEST_TENSOR_OPS_H
#define ANAKIN2_UTILS_TEST_TENSOR_OPS_H

#include "tensor.h"
#include "utils/logger/logger.h"
#include <math.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef USE_ARM_PLACE
#include <arm_neon.h>
#endif

/*
 * \brief fill tensor with constant data, only fill the cpu data
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void fill_tensor_const(anakin::saber::Tensor<Dim, target, dtype, layout>& tensor, \
    dtype value){

    dtype* data = (dtype*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<Dim> shape = tensor.get_shape();
    anakin::saber::Shape<Dim> real_shape = tensor.get_real_shape();

    for (int i = 0; i < real_shape.size(); ++i) {
        data[i] = value;
    }
}

#ifdef USE_ARM_PLACE
template<>
void fill_tensor_const(anakin::saber::Tensor<4, ARM, float, NCHW>& tensor, \
    float value){

    float* data = (float*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    data_row[l] = value;
                }
            }
        }
    }
}
#endif

#ifdef USE_CUDA
template<>
void fill_tensor_const(anakin::saber::Tensor<4, RTCUDA, float, NCHW>& tensor, \
    float value){

    float* data = (float*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    data_row[l] = value;
                }
            }
        }
    }
}
#endif

/*
 * \brief fill tensor with random data, only fill the cpu data
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void fill_tensor_rand(anakin::saber::Tensor<Dim, target, dtype, layout>& tensor, \
    dtype vstart, dtype vend){

    dtype* data = (dtype*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<Dim> real_shape = tensor.get_real_shape();

    for (int i = 0; i < real_shape.size(); ++i) {
        data[i] = vstart + (vend - vstart) * static_cast<dtype>(rand()) / RAND_MAX;
    }
}
#ifdef USE_ARM_PLACE
template<>
void fill_tensor_rand(anakin::saber::Tensor<4, ARM, float, NCHW>& tensor, \
    float vstart, float vend){

    float* data = (float*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    data_row[l] = vstart + (vend - vstart) * \
                        static_cast<float>(rand()) / RAND_MAX;
                }
            }
        }
    }
}
#endif

#ifdef USE_CUDA
template<>
void fill_tensor_rand(anakin::saber::Tensor<4, RTCUDA, float, NCHW>& tensor, \
    float vstart, float vend){

    float* data = (float*) tensor.get_cpu_data_mutable();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    data_row[l] = vstart + (vend - vstart) * \
                        static_cast<float>(rand()) / RAND_MAX;
                }
            }
        }
    }
}
#endif

/*
 * \brief sum of cpu data
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void sum_tensor_cpu(anakin::saber::Tensor<Dim, target, dtype, layout>& tensor, dtype* sum){

    dtype* data = (dtype*) tensor.get_cpu_data();
    anakin::saber::Shape<Dim> real_shape = tensor.get_real_shape();

    *sum = static_cast<dtype>(0);
    for (int i = 0; i < real_shape.size(); ++i) {
        *sum += fabs(data[i]);
    }
}

#ifdef USE_ARM_PLACE
template<>
void sum_tensor_cpu(anakin::saber::Tensor<4, ARM, float , NCHW>& tensor, float* sum){

    float* data = (float*) tensor.get_cpu_data();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    *sum = 0.f;

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    *sum += data_row[l];
                }
            }
        }
    }
}
#endif

#ifdef USE_CUDA
template<>
void sum_tensor_cpu(anakin::saber::Tensor<4, RTCUDA, float , NCHW>& tensor, float* sum){

    float* data = (float*) tensor.get_cpu_data();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    *sum = 0.f;

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch = data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel = data_batch + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row = data_channel + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    *sum += data_row[l];
                }
            }
        }
    }
}
#endif

/*
 * \brief compare differences between two tensor
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
bool tensor_diff_cpu(anakin::saber::Tensor<Dim, target, dtype, layout>& tensor1, \
    anakin::saber::Tensor<Dim, target, dtype, layout>& tensor2, dtype* diff){

    bool flag = true;

    const float eps = 1e-6;
    const float thresh = 1e-6f;

    dtype* data1 = (dtype*) tensor1.get_cpu_data();
    dtype* data2 = (dtype*) tensor2.get_cpu_data();
    anakin::saber::Shape<Dim> real_shape = tensor1.get_real_shape();
    CHECK_EQ(real_shape == tensor2.get_real_shape(), true) << "tensor size do not match";

    *diff = static_cast<dtype>(0);
    for (int i = 0; i < real_shape.size(); ++i) {
        float fdiff = fabsf(data1[i] - data2[i]);
        if (fdiff / fabsf(data1[i] + eps) > thresh){
            flag = false;
        }
        *diff += fdiff;
    }
    return flag;
}

#ifdef USE_ARM_PLACE
template<>
bool tensor_diff_cpu(anakin::saber::Tensor<4, ARM, float, NCHW>& tensor1, \
    anakin::saber::Tensor<4, ARM, float, NCHW>& tensor2, float* diff){

    bool flag = true;

    const float eps = 1e-6;
    const float thresh = 1e-6f;

    float* data1 = (float*) tensor1.get_cpu_data();
    float* data2 = (float*) tensor2.get_cpu_data();
    anakin::saber::Shape<4> shape = tensor1.get_shape();
    anakin::saber::Shape<4> real_shape1 = tensor1.get_real_shape();
    anakin::saber::Shape<4> real_shape2 = tensor2.get_real_shape();

    CHECK_EQ(shape == tensor2.get_shape(), true) << "tensor size do not match";

    int stride_row1 = real_shape1[3];
    int stride_channel1 = real_shape1.count(2, 3);
    int stride_batch1 = real_shape1.count(1, 3);

    int stride_row2 = real_shape2[3];
    int stride_channel2 = real_shape2.count(2, 3);
    int stride_batch2 = real_shape2.count(1, 3);

    *diff = 0.f;

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch1 = data1 + i * stride_batch1;
        float* data_batch2 = data2 + i * stride_batch2;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel1 = data_batch1 + j * stride_channel1;
            float* data_channel2 = data_batch2 + j * stride_channel2;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row1 = data_channel1 + k * stride_row1;
                float* data_row2 = data_channel2 + k * stride_row2;
                for (int l = 0; l < shape[3]; ++l) {
                    float fdiff = fabsf(data_row1[i] - data_row2[i]);
                    if (fdiff / fabsf(data_row1[i] + eps) > thresh){
                        flag = false;
                    }
                    *diff += fdiff;
                }
            }
        }
    }
    return flag;
}
#endif

#ifdef USE_CUDA
template<>
bool tensor_diff_cpu(anakin::saber::Tensor<4, RTCUDA, float, NCHW>& tensor1, \
    anakin::saber::Tensor<4, RTCUDA, float, NCHW>& tensor2, float* diff){

    bool flag = true;

    const float eps = 1e-6;
    const float thresh = 1e-6f;

    float* data1 = (float*) tensor1.get_cpu_data();
    float* data2 = (float*) tensor2.get_cpu_data();
    anakin::saber::Shape<4> shape = tensor1.get_shape();
    anakin::saber::Shape<4> real_shape1 = tensor1.get_real_shape();
    anakin::saber::Shape<4> real_shape2 = tensor2.get_real_shape();

    CHECK_EQ(shape == tensor2.get_shape(), true) << "tensor size do not match";

    int stride_row1 = real_shape1[3];
    int stride_channel1 = real_shape1.count(2, 3);
    int stride_batch1 = real_shape1.count(1, 3);

    int stride_row2 = real_shape2[3];
    int stride_channel2 = real_shape2.count(2, 3);
    int stride_batch2 = real_shape2.count(1, 3);

    *diff = 0.f;

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch1 = data1 + i * stride_batch1;
        float* data_batch2 = data2 + i * stride_batch2;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel1 = data_batch1 + j * stride_channel1;
            float* data_channel2 = data_batch2 + j * stride_channel2;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row1 = data_channel1 + k * stride_row1;
                float* data_row2 = data_channel2 + k * stride_row2;
                for (int l = 0; l < shape[3]; ++l) {
                    float fdiff = fabsf(data_row1[i] - data_row2[i]);
                    if (fdiff / fabsf(data_row1[i] + eps) > thresh){
                        flag = false;
                    }
                    *diff += fdiff;
                }
            }
        }
    }

    float* gpu_data1 = (float*)malloc(tensor1.capacity());
    float* gpu_data2 = (float*)malloc(tensor2.capacity());
    CHECK_EQ(gpu_data1 == nullptr, false) << "alloc mem failed";
    CHECK_EQ(gpu_data2 == nullptr, false) << "alloc mem failed";

    int offset1 = tensor1.get_offset();
    int offset2 = tensor2.get_offset();
    tensor1.set_offset(0);
    tensor2.set_offset(0);
    cudaMemcpy(gpu_data1, tensor1.get_gpu_data(), tensor1.capacity(), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_data2, tensor2.get_gpu_data(), tensor2.capacity(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch1 = gpu_data1 + i * stride_batch1;
        float* data_batch2 = gpu_data2 + i * stride_batch2;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel1 = data_batch1 + j * stride_channel1;
            float* data_channel2 = data_batch2 + j * stride_channel2;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row1 = data_channel1 + k * stride_row1;
                float* data_row2 = data_channel2 + k * stride_row2;
                for (int l = 0; l < shape[3]; ++l) {
                    float fdiff = fabsf(data_row1[i] - data_row2[i]);
                    if (fdiff / fabsf(data_row1[i] + eps) > thresh){
                        flag = false;
                    }
                    *diff += fdiff;
                }
            }
        }
    }

    tensor1.set_offset(offset1);
    tensor2.set_offset(offset2);

    free(gpu_data1);
    free(gpu_data2);

    return flag;
}

bool tensor_diff_inplace(anakin::saber::Tensor<4, RTCUDA, float, NCHW>& tensor, float* diff){

    bool flag = true;

    const float eps = 1e-6;
    const float thresh = 1e-6f;

    float* data_cpu = (float*) tensor.get_cpu_data();
    anakin::saber::Shape<4> shape = tensor.get_shape();
    anakin::saber::Shape<4> real_shape = tensor.get_real_shape();

    float* gpu_data = (float*)malloc(tensor.capacity());
    CHECK_EQ(gpu_data == nullptr, false) << "alloc mem failed";

    int offset = tensor.get_offset();
    tensor.set_offset(0);
    cudaMemcpy(gpu_data, tensor.get_gpu_data(), tensor.capacity(), cudaMemcpyDeviceToHost);

    int stride_row = real_shape[3];
    int stride_channel = real_shape.count(2, 3);
    int stride_batch = real_shape.count(1, 3);

    *diff = 0.f;

    for (int i = 0; i < shape[0]; ++i) {
        float* data_batch1 = data_cpu + i * stride_batch;
        float* data_batch2 = gpu_data + i * stride_batch;
        for (int j = 0; j < shape[1]; ++j) {
            float* data_channel1 = data_batch1 + j * stride_channel;
            float* data_channel2 = data_batch2 + j * stride_channel;
            for (int k = 0; k < shape[2]; ++k) {
                float* data_row1 = data_channel1 + k * stride_row;
                float* data_row2 = data_channel2 + k * stride_row;
                for (int l = 0; l < shape[3]; ++l) {
                    float fdiff = fabsf(data_row1[i] - data_row2[i]);
                    if (fdiff / fabsf(data_row1[i] + eps) > thresh){
                        flag = false;
                    }
                    *diff += fdiff;
                }
            }
        }
    }

    tensor.set_offset(offset);
    free(gpu_data);

    return flag;
}
#endif

#endif //ANAKIN2_UTILS_TEST_TENSOR_OPS_H
