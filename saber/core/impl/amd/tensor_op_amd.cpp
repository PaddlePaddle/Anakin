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

#ifdef AMD_GPU

#include <random>
#include "utils/amd_logger.h"

namespace anakin {

namespace saber {

typedef TargetWrapper<AMD> AMD_API;

#if 1

template <typename Dtype>
void fill_tensor_const_impl(
    cl_command_queue stream,
    cl_mem dio,
    Dtype value,
    size_t size,
    size_t offset) {

    cl_event event;
    clEnqueueFillBuffer(stream, dio, &value, sizeof(Dtype), offset, size, 0, NULL, &event);
    clFlush(stream);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}

template <typename TargetType>
void fill_tensor_const(
    Tensor<TargetType>& tensor,
    float value,
    typename Tensor<TargetType>::API::stream_t stream) {

    cl_mem dio      = (cl_mem)tensor.mutable_data();
    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if (stream == nullptr) {
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    size_t dsize   = tensor.valid_size() * tensor.get_dtype_size();
    size_t doffset = tensor.data_offset() * tensor.get_dtype_size();
    DataType type  = tensor.get_dtype();

    switch (type) {
    case AK_UINT8:
        fill_tensor_const_impl(stream, dio, static_cast<unsigned char>(value), dsize, doffset);
        break;

    case AK_INT8:
        fill_tensor_const_impl(stream, dio, static_cast<char>(value), dsize, doffset);
        break;

    case AK_INT16:
        fill_tensor_const_impl(stream, dio, static_cast<short>(value), dsize, doffset);
        break;

    case AK_UINT16:
        fill_tensor_const_impl(stream, dio, static_cast<unsigned short>(value), dsize, doffset);
        break;

    case AK_HALF:
        fill_tensor_const_impl(stream, dio, static_cast<short>(value), dsize, doffset);
        break;

    case AK_UINT32:
        fill_tensor_const_impl(stream, dio, static_cast<unsigned int>(value), dsize, doffset);
        break;

    case AK_INT32:
        fill_tensor_const_impl(stream, dio, static_cast<int>(value), dsize, doffset);
        break;

    case AK_FLOAT:
        fill_tensor_const_impl(stream, dio, static_cast<float>(value), dsize, doffset);
        break;

    case AK_DOUBLE:
        fill_tensor_const_impl(stream, dio, static_cast<double>(value), dsize, doffset);
        break;

    default:
        LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
};

template <typename Dtype>
void fill_tensor_rand_impl_1(Dtype* dio, size_t size) {
    for (int i = 0; i < size; ++i) {
        dio[i] = static_cast<Dtype>(rand());
    }
}

template <typename TargetType>
void fill_tensor_rand(
    Tensor<TargetType>& tensor,
    typename Tensor<TargetType>::API::stream_t stream) {

    typedef typename TargetWrapper<TargetType>::TPtr PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem   = (cl_mem)ptr;

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if (stream == nullptr) {
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    int size       = tensor.valid_size();
    size_t dsize   = size * tensor.get_dtype_size();
    size_t doffset = tensor.data_offset() * tensor.get_dtype_size();
    void* dio      = (void*)malloc(dsize);

    DataType type = tensor.get_dtype();

    switch (type) {
    case AK_UINT8:
        fill_tensor_rand_impl_1(reinterpret_cast<unsigned char*>(dio), size);
        break;

    case AK_INT8:
        fill_tensor_rand_impl_1(reinterpret_cast<char*>(dio), size);
        break;

    case AK_INT16:
        fill_tensor_rand_impl_1(reinterpret_cast<short*>(dio), size);
        break;

    case AK_UINT16:
        fill_tensor_rand_impl_1(reinterpret_cast<unsigned short*>(dio), size);
        break;

    case AK_HALF:
        fill_tensor_rand_impl_1(reinterpret_cast<short*>(dio), size);
        break;

    case AK_UINT32:
        fill_tensor_rand_impl_1(reinterpret_cast<unsigned int*>(dio), size);
        break;

    case AK_INT32:
        fill_tensor_rand_impl_1(reinterpret_cast<int*>(dio), size);
        break;

    case AK_FLOAT:
        fill_tensor_rand_impl_1(reinterpret_cast<float*>(dio), size);
        break;

    case AK_DOUBLE:
        fill_tensor_rand_impl_1(reinterpret_cast<double*>(dio), size);
        break;

    default:
        LOG(FATAL) << "data type: " << type << " is unsupported now";
    }

    ALOGD("mem = " << mem << " size=" << dsize);
    cl_int err = clEnqueueWriteBuffer(stream, mem, CL_TRUE, doffset, dsize, dio, 0, NULL, NULL);
    free(dio);

    if (err != CL_SUCCESS) {
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }
};

template <typename Dtype>
void fill_tensor_rand_impl_2(Dtype* dio, size_t size, float vstart, float vend) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);

    for (int i = 0; i < size; ++i) {
        Dtype random_num = vstart + (vend - vstart) * dis(gen);
        dio[i]           = random_num;
    }
}
template <typename TargetType>
void fill_tensor_rand(
    Tensor<TargetType>& tensor,
    float vstart,
    float vend,
    typename Tensor<TargetType>::API::stream_t stream) {

    typedef typename TargetWrapper<TargetType>::TPtr PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem   = (cl_mem)ptr;

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if (stream == nullptr) {
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    int size       = tensor.valid_size();
    size_t dsize   = size * tensor.get_dtype_size();
    size_t doffset = tensor.data_offset() * tensor.get_dtype_size();
    void* dio      = (void*)malloc(dsize);

    DataType type = tensor.get_dtype();

    switch (type) {
    case AK_UINT8:
        fill_tensor_rand_impl_2(reinterpret_cast<unsigned char*>(dio), size, vstart, vend);
        break;

    case AK_INT8:
        fill_tensor_rand_impl_2(reinterpret_cast<char*>(dio), size, vstart, vend);
        break;

    case AK_INT16:
        fill_tensor_rand_impl_2(reinterpret_cast<short*>(dio), size, vstart, vend);
        break;

    case AK_UINT16:
        fill_tensor_rand_impl_2(reinterpret_cast<unsigned short*>(dio), size, vstart, vend);
        break;

    case AK_HALF:
        fill_tensor_rand_impl_2(reinterpret_cast<short*>(dio), size, vstart, vend);
        break;

    case AK_UINT32:
        fill_tensor_rand_impl_2(reinterpret_cast<unsigned int*>(dio), size, vstart, vend);
        break;

    case AK_INT32:
        fill_tensor_rand_impl_2(reinterpret_cast<int*>(dio), size, vstart, vend);
        break;

    case AK_FLOAT:
        fill_tensor_rand_impl_2(reinterpret_cast<float*>(dio), size, vstart, vend);
        break;

    case AK_DOUBLE:
        fill_tensor_rand_impl_2(reinterpret_cast<double*>(dio), size, vstart, vend);
        break;

    default:
        LOG(FATAL) << "data type: " << type << " is unsupported now";
    }

    ALOGD("mem = " << mem << " size=" << dsize);
    cl_int err = clEnqueueWriteBuffer(stream, mem, CL_TRUE, doffset, dsize, dio, 0, NULL, NULL);
    free(dio);

    if (err != CL_SUCCESS) {
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }
};

template <typename Dtype>
print_tensor_impl(Dtype* dio, size_t size, size_t width) {
    std::string tmp("\n");

    for (int i = 0; i < size; ++i) {
        tmp.append(std::to_string(static_cast<float>(dio[i])) + " ");

        if ((i + 1) % width == 0) {
            tmp.append("\n");
        }
    }

    ALOGI(tmp.c_str());
}

template <typename TargetType>
void print_tensor(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    typedef typename TargetWrapper<TargetType>::TPtr PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem   = (cl_mem)ptr;

    LOG(INFO) << "device tensor size: " << tensor.valid_size()
              << " type size: " << tensor.get_dtype_size() << " mem: " << mem;

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if (stream == nullptr) {
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    int size       = tensor.valid_size();
    size_t dsize   = size * tensor.get_dtype_size();
    size_t doffset = tensor.data_offset() * tensor.get_dtype_size();
    void* dio      = (void*)malloc(dsize);

    cl_int err = clEnqueueReadBuffer(stream, mem, CL_TRUE, doffset, dsize, dio, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        free(dio);
        return;
    }

    DataType type = tensor.get_dtype();

    switch (type) {
    case AK_UINT8:
        print_tensor_impl(reinterpret_cast<unsigned char*>(dio), size, tensor.width());
        break;

    case AK_INT8:
        print_tensor_impl(reinterpret_cast<char*>(dio), size, tensor.width());
        break;

    case AK_INT16:
        print_tensor_impl(reinterpret_cast<short*>(dio), size, tensor.width());
        break;

    case AK_UINT16:
        print_tensor_impl(reinterpret_cast<unsigned short*>(dio), size, tensor.width());
        break;

    case AK_HALF:
        print_tensor_impl(reinterpret_cast<short*>(dio), size, tensor.width());
        break;

    case AK_UINT32:
        print_tensor_impl(reinterpret_cast<unsigned int*>(dio), size, tensor.width());
        break;

    case AK_INT32:
        print_tensor_impl(reinterpret_cast<int*>(dio), size, tensor.width());
        break;

    case AK_FLOAT:
        print_tensor_impl(reinterpret_cast<float*>(dio), size, tensor.width());
        break;

    case AK_DOUBLE:
        print_tensor_impl(reinterpret_cast<double*>(dio), size, tensor.width());
        break;

    default:
        LOG(FATAL) << "data type: " << type << " is unsupported now";
    }

    free(dio);
    // clFinish(stream);
};


template<>
double tensor_mean_value<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}

template<>
double tensor_mean_value_valid<AMD>(Tensor<AMD>& tensor,
                                   typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}

template void
print_tensor<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL);
template void fill_tensor_const<AMD>(
    Tensor<AMD>& tensor,
    float value,
    typename Tensor<AMD>::API::stream_t stream = NULL);
template void
fill_tensor_rand<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream);
template void fill_tensor_rand<AMD>(
    Tensor<AMD>& tensor,
    float vstart,
    float vend,
    typename Tensor<AMD>::API::stream_t stream);

#endif
} // namespace saber

} // namespace anakin

#endif // AMD_GPU
