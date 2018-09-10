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

#ifndef ANAKIN_SABER_FUNCS_DEBUG_H
#define ANAKIN_SABER_FUNCS_DEBUG_H

#include "tensor.h"
namespace anakin {
namespace saber {

template <typename Target_Type>
struct DefaultHostType {
    typedef X86 Host_type;
};

template <>
struct DefaultHostType<NV> {
    typedef NVHX86 Host_type;
};

template <>
struct DefaultHostType<ARM> {
    typedef ARM Host_type;
};


template <typename Target_Type>
static void write_tensorfile(Tensor<Target_Type>& tensor, const char* locate) {

    typedef typename DefaultHostType<Target_Type>::Host_type HOST_TYPE;
    Tensor<HOST_TYPE> host_tensor;
    host_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
    host_tensor.copy_from(tensor);
    LOG(INFO) << "target tensor data:" << tensor.size();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open field " << locate;

    } else {
        if (tensor.get_dtype() == AK_FLOAT) {
            const float* data_ptr = (const float*)host_tensor.data();
            int size = host_tensor.valid_size();

            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %g \n", i, (data_ptr[i]));
            }
        } else {
            LOG(FATAL) << "not supported write type";
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}

template <typename TargetType>
static void record_dev_tensorfile(const float* dev_tensor, int size, const char* locate) {};

#ifdef USE_CUDA
template <>
void record_dev_tensorfile<NV>(const float* dev_tensor, int size, const char* locate) {
    Tensor <NVHX86> host_temp;
    host_temp.re_alloc(Shape({1, 1, 1, size}, Layout_NCHW), AK_FLOAT);
    CUDA_CHECK(cudaMemcpy(host_temp.mutable_data(), dev_tensor, sizeof(float) * size,
                          cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {
        const float* data = (const float*)host_temp.data();

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (data[i]));
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}
static void record_dev_tensorfile(Tensor <NV>* dev_tensor, const char* locate) {
    Tensor <NVHX86> host_temp;

    int size = dev_tensor->valid_size();
    host_temp.re_alloc(Shape({1, 1, 1, size}, Layout_NCHW), dev_tensor->get_dtype());
    CUDA_CHECK(cudaMemcpy(host_temp.mutable_data(), dev_tensor->data(), sizeof(float) * size,
                          cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {
        const float* data = (const float*)host_temp.data();

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (data[i]));
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}
#endif

#ifdef USE_X86_PLACE
template<>
void record_dev_tensorfile<X86>(const float* dev_tensor, int size, const char* locate) {
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (dev_tensor[i]));
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}
static void record_dev_tensorfile(Tensor <X86>* dev_tensor, const char* locate) {
    int size = dev_tensor->valid_size();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (((float*)dev_tensor->data())[i]));
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}
#endif

#if defined(USE_X86_PLACE) || defined(USE_CUDA)
template <typename HTensor>
static void readTensorData(HTensor tensor, const char* locate) {
    FILE* fp = fopen(locate, "rb");

    if (fp == 0) {
        CHECK(false) << "file open failed " << locate;

    } else {
        LOG(INFO) << "file open success [" << locate << " ],read " << tensor.valid_shape().count();
        size_t size = fread(tensor.mutable_data(), sizeof(float), tensor.valid_size(), fp);
        CHECK_EQ(size, tensor.valid_shape().count()) << "read data file [" << locate << "], size not match";
        fclose(fp);
    }
}

#endif
}
}



#endif //ANAKIN_DEBUG_H
