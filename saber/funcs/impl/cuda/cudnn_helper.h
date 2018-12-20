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

#ifndef ANAKIN_SABER_CUDNN_HELPER_H
#define ANAKIN_SABER_CUDNN_HELPER_H

#include "anakin_config.h"
#include "saber/core/common.h"
#include "saber/saber_types.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace anakin{

namespace cudnn{

struct ParamsRegion {

    ParamsRegion():_offset(NULL), _size(0){};
    ParamsRegion(void *offset, size_t size):_offset(offset), _size(size){}
    ~ParamsRegion(){}
    ParamsRegion(const ParamsRegion &right): _offset(right._offset),_size(right._size){};

    ParamsRegion &operator=(const ParamsRegion &right) {
        _offset = right._offset;
        _size=right._size;
        return *this;
    }
    bool operator==(const ParamsRegion &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_offset == right._offset);
        comp_eq = comp_eq && (_size == right._size);
        return  comp_eq;
    }

    void * _offset;
    size_t _size;
};


template <typename T>
class cudnnTypeWrapper;

// some cudnn op descriptor type is not same as data type
template <typename T>
class cudnnOpWrapper;

template <>
class cudnnOpWrapper<char> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_INT32;
};

template <>
class cudnnOpWrapper<float> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template <>
class cudnnTypeWrapper<float> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    typedef const float ScalingParamType;
    typedef float BNParamType;
    static ScalingParamType* kOne() {
        static ScalingParamType v = 1.0;
        return &v;
    }
    static const ScalingParamType* kZero() {
        static ScalingParamType v = 0.0;
        return &v;
    }
};

template <>
class cudnnTypeWrapper<double> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    typedef const double ScalingParamType;
    typedef double BNParamType;
    static ScalingParamType* kOne() {
        static ScalingParamType v = 1.0;
        return &v;
    }
    static ScalingParamType* kZero() {
        static ScalingParamType v = 0.0;
        return &v;
    }
};

template <typename T>
class TensorDescriptors {
public:
    TensorDescriptors(
            size_t n,
            const std::vector<std::vector<int>>& dim,
            const std::vector<std::vector<int>>& stride) {
        descs_.resize(n);
        CHECK_EQ(dim.size(), stride.size());
        for (auto i = 0; i < n; ++i) {
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs_[i]));
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(
                    descs_[i],
                    cudnnTypeWrapper<T>::type,
                    dim[i].size(),
                    dim[i].data(),
                    stride[i].data()));
        }
    }
    ~TensorDescriptors() {
        for (auto desc : descs_) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
        }
    }
    const cudnnTensorDescriptor_t* descs() const {
        return descs_.data();
    }
    const int sizes() const {
        return descs_.size();
    }
private:
    std::vector<cudnnTensorDescriptor_t> descs_;
};

template <>
class cudnnTypeWrapper<char> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_INT8;
    typedef const char ScalingParamType;
    typedef double BNParamType;
    static ScalingParamType* kOne() {
        static ScalingParamType v = 1;
        return &v;
    }
    static ScalingParamType* kZero() {
        static ScalingParamType v = 0;
        return &v;
    }
};

template <typename Dtype>
inline void createTensorDesc(cudnnTensorDescriptor_t* desc) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    int nbDims, int dimA[], int strideA[]) {
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc,
                    cudnnTypeWrapper<Dtype>::type,
                    nbDims,
                    dimA,
                    strideA));
}
template <typename Dtype>
inline void setRNNDesc(cudnnRNNDescriptor_t* rnnDesc,
                       cudnnHandle_t cudnnHandle,
                       int hiddenSize, int numLayers,
                       cudnnDropoutDescriptor_t dropoutDesc,
                       int numDirection,
                       cudnnRNNMode_t mode
)
{
#if CUDNN_MAJOR >= 6
    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                      	 *rnnDesc,
                                      	 hiddenSize,
                                      	 numLayers,
                                      	 dropoutDesc,
                                      	 CUDNN_LINEAR_INPUT,
                                      	 numDirection ==2? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                      	 mode,
                                      	 CUDNN_RNN_ALGO_STANDARD,
                                      	 cudnnTypeWrapper<Dtype>::type ));
#else
    CUDNN_CHECK(cudnnSetRNNDescriptor(*rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      dropoutDesc,
                                      CUDNN_LINEAR_INPUT,
                                      numDirection ==2? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                      mode,
                                      cudnnTypeWrapper<Dtype>::type ));
#endif
}
template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
                            cudnnTensorFormat_t format, 
                            int n, int c, int h, int w) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc,
                    format,
                    cudnnTypeWrapper<Dtype>::type,
                    n, c, h, w));

}

template <typename Dtype>
inline void setTensor4dDescEx(cudnnTensorDescriptor_t* desc,
                            int n, int c, int h, int w, 
                            int n_stride, 
                            int c_stride,
                            int h_stride,
                            int w_stride) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc,
                    cudnnTypeWrapper<Dtype>::type,
                    n, c, h, w,
                    n_stride, c_stride, h_stride, w_stride));

}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc) {
    CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

template <typename Dtype>
inline void setNDFilterDesc(cudnnFilterDescriptor_t* desc, int nDim, int dim[],
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(*desc, cudnnTypeWrapper<Dtype>::type,
                    format, nDim, dim));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionNdDesc(cudnnConvolutionDescriptor_t* conv,
    int arrayLength, int padA[], int filterStrideA[], int dilationA[]) {

    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(*conv,
                    arrayLength, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION,
                    cudnnOpWrapper<Dtype>::type));
}

template <typename Dtype>
inline void set_math_type(cudnnConvolutionDescriptor_t* conv, bool use_tensor_core) {
#if CUDNN_VERSION_MIN(7, 0, 0)
    cudnnMathType_t math_type = use_tensor_core ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
    CUDNN_CHECK(cudnnSetConvolutionMathType(*conv, math_type));

#endif
}

template <typename Dtype>
inline void set_group_count(cudnnConvolutionDescriptor_t* conv, int group) {
#if CUDNN_VERSION_MIN(7, 0, 0)
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(*conv, group));
#endif
}

template <typename Dtype>
inline void create_activation_des(cudnnActivationDescriptor_t* desc) {

    CUDNN_CHECK(cudnnCreateActivationDescriptor(desc));
}

template <typename Dtype>
inline void set_activation_des(cudnnActivationDescriptor_t *desc, saber::ActiveType act,
                Dtype clip_threadhold = Dtype(0)) {

    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
    switch (act){
        case saber::Active_sigmoid:
            mode = CUDNN_ACTIVATION_SIGMOID; break;
        case saber::Active_relu:
            mode = CUDNN_ACTIVATION_RELU; break;
        case saber::Active_tanh:
            mode = CUDNN_ACTIVATION_TANH; break;
        case saber::Active_clipped_relu:
            mode = CUDNN_ACTIVATION_CLIPPED_RELU; break;
        case saber::Active_elu:
            mode = CUDNN_ACTIVATION_ELU; break;
        default:
            LOG(FATAL)<<"error in activeType!!!"; break;
    }

    CUDNN_CHECK(cudnnSetActivationDescriptor(*desc,
                    mode, CUDNN_NOT_PROPAGATE_NAN, clip_threadhold));
}

template <typename Dtype>
inline void create_pooling_des(cudnnPoolingDescriptor_t * desc) {

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(desc));
}

template <typename Dtype>
inline void set_nd_pooling_des(cudnnPoolingDescriptor_t* pooling, saber::PoolingType pooling_type,
                                int nbDims, int windowDImA[], int paddingA[], int strideA[]) {

    cudnnPoolingMode_t mode;
    switch (pooling_type){
        case saber::Pooling_max:
            mode = CUDNN_POOLING_MAX; break;
        case saber::Pooling_average_include_padding:
            mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
        case saber::Pooling_average_exclude_padding:
            mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
        case saber::Pooling_max_deterministic:
            mode = CUDNN_POOLING_MAX_DETERMINISTIC; break;
        default:
            LOG(FATAL)<<"error in poolingType!!!!"; break;
    }
    CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pooling, mode, CUDNN_NOT_PROPAGATE_NAN,
                    nbDims, windowDImA, paddingA, strideA));
//    CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*conv, group));
}
template <typename Dtype>
inline void createLrnDesc(cudnnLRNDescriptor_t * desc) {

    CUDNN_CHECK(cudnnCreateLRNDescriptor(desc));
}
template <typename Dtype>
inline void setLrnDesc(cudnnLRNDescriptor_t* desc,
                        int N,
                        float alpha,
                        float beta,
                        float K) {
    CUDNN_CHECK(cudnnSetLRNDescriptor(*desc, N, alpha, beta, K));
}

} // namespace saber
} // namespace anakin

#endif //SABER_CUDNN_HELPER_H
