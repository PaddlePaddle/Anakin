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

static void force_reformat_locate_file_path(const char* locate){
    char* path= const_cast<char *>(locate);
    for(int i =0;path[i]!='\0';i++){
        if(path[i]=='/'){
            path[i]='_';
        }

    }
}
#if defined(USE_X86_PLACE)


static void record_tensor_to_file(Tensor <X86, AK_FLOAT, NCHW>& dev_tensor,const char* locate){
    Tensor <X86, AK_FLOAT, NCHW> host_temp;
    host_temp.re_alloc(dev_tensor.valid_shape());
    host_temp.copy_from(dev_tensor);
    force_reformat_locate_file_path(locate);
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        CHECK(false) << "file open failed " << locate;

    } else {
        for (int i = 0; i < host_temp.valid_size(); ++i) {
            fprintf(fp, "[%d] %g \n", i, (host_temp.data()[i]));
        }

        fclose(fp);
        LOG(INFO) << "!!! write success: " << locate;
    }
}
static void write_tensorfile(Tensor <X86, AK_FLOAT, NCHW> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    force_reformat_locate_file_path(locate);
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open field " << locate;

    } else {
        const Dtype* data_ptr = static_cast<const Dtype*>(tensor.data());
        int size = tensor.valid_size();

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (data_ptr[i]));
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}
#endif


#ifdef USE_CUDA

static void record_tensor_to_file(Tensor <NV, AK_FLOAT, NCHW>& dev_tensor,const char* locate){
    Tensor <X86, AK_FLOAT, NCHW> host_temp;
    host_temp.re_alloc(dev_tensor.valid_shape());
    host_temp.copy_from(dev_tensor);
    force_reformat_locate_file_path(locate);
    FILE* fp = fopen(locate, "w+");
    cudaDeviceSynchronize();
    if (fp == 0) {
        CHECK(false) << "file open failed " << locate;

    } else {
        for (int i = 0; i < host_temp.valid_size(); ++i) {
            fprintf(fp, "[%d] %g \n", i, (host_temp.data()[i]));
        }

        fclose(fp);
                LOG(INFO) << "!!! write success: " << locate;
    }
}


static void record_dev_tensorfile(Tensor <NV, AK_FLOAT, NCHW>* dev_tensor, const char* locate) {
    Tensor <X86, AK_FLOAT, NCHW> host_temp;
    int size=dev_tensor->valid_size();
    host_temp.re_alloc(Shape(1, 1, 1, size));
    CUDA_CHECK(cudaMemcpy(host_temp.mutable_data(), dev_tensor->data(), sizeof(float) * size,
                          cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
                LOG(ERROR) << "file open failed " << locate;

    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (host_temp.data()[i]));
        }

        fclose(fp);
    }

            LOG(INFO) << "!!! write success: " << locate;
}

#endif

#ifdef USE_X86_PLACE

static void record_dev_tensorfile(Tensor <X86, AK_FLOAT, NCHW>* dev_tensor, const char* locate) {
    int size=dev_tensor->valid_size();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %g \n", i, (dev_tensor->data()[i]));
        }

        fclose(fp);
    }

        LOG(INFO) << "!!! write success: " << locate;
}
#endif

#if defined(USE_X86_PLACE) || defined(USE_CUDA)
static void readTensorData(Tensor<X86, AK_FLOAT, NCHW> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    FILE* fp = fopen(locate, "rb");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;
        exit(0);

    } else {
        LOG(INFO) << "file open success [" << locate << " ],read " << tensor.valid_shape().count();
        size_t size=fread(tensor.mutable_data(), sizeof(Dtype), tensor.valid_size(), fp);
        CHECK_EQ(size,tensor.valid_shape().count())<<"read data file ["<<locate<<"], size not match";
        fclose(fp);
    }
}

static void readTensorData(Tensor<X86, AK_FLOAT, NCHW_C16> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    FILE* fp = fopen(locate, "rb");

    if (fp == 0) {
                LOG(ERROR) << "file open failed " << locate;

    } else {
                LOG(INFO) << "file open success [" << locate << " ],read " << tensor.valid_shape().count();
        size_t size=fread(tensor.mutable_data(), sizeof(Dtype), tensor.valid_size(), fp);
        CHECK_EQ(size,tensor.valid_shape().count())<<"read data file ["<<locate<<"], size not match";
        fclose(fp);
    }
}
#endif
}
}



#endif //ANAKIN_DEBUG_H
