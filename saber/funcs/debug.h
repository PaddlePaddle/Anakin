//
// Created by Liu,Junjie(SYS) on 2018/5/28.
//

#ifndef ANAKIN_SABER_FUNCS_DEBUG_H
#define ANAKIN_SABER_FUNCS_DEBUG_H

#include "tensor.h"
namespace anakin {
namespace saber {

#if defined(USE_X86_PLACE) || defined(USE_CUDA)
static void write_tensorfile(Tensor <X86, AK_FLOAT, NCHW> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
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
static void record_dev_tensorfile(const float* dev_tensor, int size, const char* locate) {
    Tensor <X86, AK_FLOAT, NCHW> host_temp;
    host_temp.re_alloc(Shape(1, 1, 1, size));
    CUDA_CHECK(cudaMemcpy(host_temp.mutable_data(), dev_tensor, sizeof(float) * size,
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

#if defined(USE_X86_PLACE) || defined(USE_CUDA)
static void readTensorData(Tensor<X86, AK_FLOAT, NCHW> tensor, const char* locate) {
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
