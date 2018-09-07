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



#include "saber/core/tensor_op.h"
#include <limits>

#ifdef USE_AMD

#include <random>

namespace anakin{

namespace saber{

typedef TargetWrapper<AMD> AMD_API;

#if 1 
template <typename TargetType>
void fill_tensor_const(Tensor<TargetType>& tensor, \
    float value, typename Tensor<TargetType>::API::stream_t stream){

    typedef typename DataTrait<AMD, AK_FLOAT>::Dtype Dtype;
    typedef typename DataTrait<AMD, AK_FLOAT>::PtrDtype PtrDtype;

    PtrDtype data_ptr = (PtrDtype)tensor.mutable_data();
    int size = tensor.valid_size();
   
    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if(stream == nullptr){
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    cl_mem mem = (cl_mem)data_ptr;
    cl_event event;
    clEnqueueFillBuffer(stream, mem, &value, sizeof(Dtype), 0, size * sizeof(Dtype), 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
    
};


template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    typedef typename DataTrait<AMD, AK_FLOAT>::Dtype Dtype;
    typedef typename DataTrait<AMD, AK_FLOAT>::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem = (cl_mem)ptr;
    int size = tensor.valid_size();

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];
    if(stream == nullptr){
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    cl_int err;
    Dtype* data_ptr = (Dtype *)clEnqueueMapBuffer(stream, mem, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(Dtype), 0, NULL, NULL, &err);
    if (err != CL_SUCCESS){
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = static_cast<Dtype>(rand());
    }
  
    cl_event event;
    clEnqueueUnmapMemObject(stream, mem, data_ptr, 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
   
};

template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, float vstart, \
    float vend, typename Tensor<TargetType>::API::stream_t stream) {

    typedef typename DataTrait<AMD, AK_FLOAT>::Dtype Dtype;
    typedef typename DataTrait<AMD, AK_FLOAT>::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem = (cl_mem)ptr;

    int size = tensor.size();

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];
    if(stream == nullptr){
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    cl_int err;
    Dtype* data_ptr = (Dtype *)clEnqueueMapBuffer(stream, mem, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(Dtype), 0, NULL, NULL, &err);
    if (err != CL_SUCCESS){
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    for (int i = 0; i < size; ++i) {
        Dtype random_num = vstart + (vend - vstart) * dis(gen);
        data_ptr[i] = random_num;
    }
  
    cl_event event;
    clEnqueueUnmapMemObject(stream, mem, data_ptr, 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
};

template <typename TargetType>
void print_tensor(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream){

    typedef typename DataTrait<AMD, AK_FLOAT>::Dtype Dtype;
    typedef typename DataTrait<AMD, AK_FLOAT>::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem = (cl_mem)ptr;

    LOG(INFO) << "device tensor size: " << tensor.size() << " type size: " << sizeof(Dtype);
    int size = tensor.size();

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if(stream == nullptr){
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    cl_int err;
    Dtype * data_ptr = (Dtype *)clEnqueueMapBuffer(stream, mem, CL_TRUE, CL_MAP_READ, 0, size * sizeof(Dtype), 0, NULL, NULL, &err);
    if (err != CL_SUCCESS){
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }

    for (int i = 0; i < size; ++i) {
        printf("%.5f ", static_cast<float>(data_ptr[i]));
        if ((i + 1) % tensor.width() == 0) {
            printf("\n");
        }
    }
    printf("\n");
  
    clEnqueueUnmapMemObject(stream, mem, data_ptr, 0, NULL, NULL);
    //clFinish(stream);

};

template void print_tensor<AMD> \
    (Tensor<AMD>& tensor, \
    typename Tensor<AMD>::API::stream_t stream = NULL);
template void fill_tensor_const<AMD>\
    (Tensor<AMD>& tensor, \
    float value, typename Tensor<AMD>::API::stream_t stream = NULL); 
template void fill_tensor_rand<AMD>\
     (Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream);
template void fill_tensor_rand<AMD>\
     (Tensor<AMD>& tensor, float vstart, \
    float vend, typename Tensor<AMD>::API::stream_t stream);

#endif
} //namespace saber

} //namespace anakin

#endif //USE_AMD
