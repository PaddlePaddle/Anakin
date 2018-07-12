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
template <class Tensor_t>
void fill_tensor_device_const(Tensor_t& tensor, \
    typename Tensor_t::Dtype value, \
    typename Tensor_t::API::stream_t stream){

    typedef typename Tensor_t::Dtype Dtype;
    typedef typename Tensor_t::PtrDtype PtrDtype;

    PtrDtype data_ptr = (PtrDtype)tensor.get_buf()->get_data_mutable();
    int size = tensor.size();
   
    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if(stream == nullptr){
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }


    cl_mem mem = data_ptr.dmem;
    cl_event event;
    clEnqueueFillBuffer(stream, mem, &value, sizeof(Dtype), 0, size * sizeof(Dtype), 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
    
};


template <class Tensor_t>
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::Dtype Dtype;
    typedef typename Tensor_t::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.get_buf()->get_data_mutable();
    cl_mem mem = ptr.dmem;
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

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = static_cast<Dtype>(rand());
    }
  
    cl_event event;
    clEnqueueUnmapMemObject(stream, mem, data_ptr, 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
   
};

template <class Tensor_t>
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::Dtype vstart, \
    typename Tensor_t::Dtype vend, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::Dtype Dtype;
    typedef typename Tensor_t::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.get_buf()->get_data_mutable();
    cl_mem mem = ptr.dmem;

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

template <class Tensor_t>
void print_tensor_device(Tensor_t& tensor, typename Tensor_t::API::stream_t stream){

    typedef typename Tensor_t::Dtype Dtype;
    typedef typename Tensor_t::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.get_buf()->get_data_mutable();
    cl_mem mem = ptr.dmem;

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

#define FILL_TENSOR_AMD(type, layout) \
    template void fill_tensor_device_const<Tensor<AMD, type, layout>>\
        (Tensor<AMD, type, layout>& tensor, DataTrait<AMD, type>::Dtype value, \
        typename TargetWrapper<AMD>::stream_t stream); \
    template void fill_tensor_device_rand<Tensor<AMD, type, layout>>\
        (Tensor<AMD, type, layout>& tensor, typename TargetWrapper<AMD>::stream_t stream); \
    template void fill_tensor_device_rand<Tensor<AMD, type, layout>>\
        (Tensor<AMD, type, layout>& tensor, DataTrait<AMD, type>::Dtype vstart, \
        DataTrait<AMD, type>::Dtype vend, typename TargetWrapper<AMD>::stream_t stream); \
    template void print_tensor_device<Tensor<AMD, type, layout>>\
        (Tensor<AMD, type, layout>& tensor, typename TargetWrapper<AMD>::stream_t stream);

FILL_TENSOR_AMD(AK_FLOAT, NCHW);
FILL_TENSOR_AMD(AK_FLOAT, NHWC);
FILL_TENSOR_AMD(AK_FLOAT, NHW);
FILL_TENSOR_AMD(AK_FLOAT, NW);
FILL_TENSOR_AMD(AK_FLOAT, HW);
FILL_TENSOR_AMD(AK_FLOAT, W);

FILL_TENSOR_AMD(AK_INT8, NCHW);
FILL_TENSOR_AMD(AK_INT8, NHWC);
FILL_TENSOR_AMD(AK_INT8, NHW);
FILL_TENSOR_AMD(AK_INT8, NW);
FILL_TENSOR_AMD(AK_INT8, HW);
FILL_TENSOR_AMD(AK_INT8, W);

#endif
} //namespace saber

} //namespace anakin

#endif //USE_AMD
