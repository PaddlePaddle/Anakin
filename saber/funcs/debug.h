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
static void write_tensorfile(const Tensor<Target_Type>& tensor, const char* locate) {

    typedef typename DefaultHostType<Target_Type>::Host_type HOST_TYPE;
    Tensor<HOST_TYPE> host_tensor;
    host_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
    host_tensor.copy_from(tensor);
    LOG(INFO) << "target tensor data:" << tensor.valid_size();
    FILE* fp = fopen(locate, "w+");

    if (fp == nullptr) {
        LOG(ERROR) << "file open field " << locate;
    } else {
        if (tensor.get_dtype() == AK_FLOAT) {
            const float* data_ptr = (const float*)host_tensor.data();
            int size = host_tensor.valid_size();
            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %f \n", i, (data_ptr[i]));
            }
        } else if (tensor.get_dtype() == AK_INT8){
            const char* data_ptr = (const char*)host_tensor.data();
            int size = host_tensor.valid_size();
            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %d \n", i, (data_ptr[i]));
            }
        } else {
            LOG(FATAL) << "not supported write type";
        }
        fclose(fp);
    }
    LOG(INFO) << "!!! write success: " << locate;
}

template <typename Target_Type>
static void record_tensor_in_format(const Tensor<Target_Type>& tensor,
        const std::string &op_type, const std::string &op_name, bool is_out, int index) {
    std::string path = "record+" + op_type +
            "+" + op_name +
            "+" + (is_out ? "out" : "in") +
            "+" + std::to_string(index) + "+";
    for (auto x : tensor.valid_shape()) {
        path += std::to_string(x) + "_";
    }
    write_tensorfile(tensor,(path+".txt").c_str());
}

template <typename Dtype>
static std::string vector_2_string(std::vector<Dtype> vec){
    std::string ans="[";
    for (auto a : vec){
        ans+=std::to_string(a)+",";
    }
    ans+="]";
    return ans;
}

}
}

#endif //ANAKIN_DEBUG_H
