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

#include "anakin_config.h"

#ifndef USE_SGX

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

template <typename HostType>
static void reorder_nchwc8_nchw(Tensor<HostType>& input,
                                Tensor<HostType>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = output.valid_shape();
    int n_value = shape[0];
    int c_value = shape[1];
    int h_value = shape[2];
    int w_value = shape[3];
    Shape shape_input = input.valid_shape();
    int c_round_div8 = shape_input[1];

    if (input.get_layout() == Layout_NCHW_C8R) {
        c_round_div8 = (shape_input.channel() + 7) / 8;
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(4) schedule(static)

    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                //#pragma ivdep
                for (int w = 0; w < w_value; ++w) {
                    int round_c = c / 8;
                    int remainder_c = c % 8;
                    int input_idx = n * c_round_div8 * h_value * w_value * 8 + round_c * h_value * w_value * 8 +
                                    h * w_value * 8 + w * 8 + remainder_c;
                    int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                     h * w_value  + w ;

                    *(output_ptr + output_idx) = input_ptr[input_idx];
                }
            }
        }
    }

}

template <typename Target_Type>
static void write_tensorfile(const Tensor<Target_Type>& tensor, const char* locate) {

    typedef typename DefaultHostType<Target_Type>::Host_type HOST_TYPE;
    Tensor<HOST_TYPE> host_tensor;
    host_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
    host_tensor.copy_from(tensor);

    if (host_tensor.get_layout() == Layout_NCHW_C8R) {
        Tensor<HOST_TYPE> temp_tensor(host_tensor.valid_shape());
        temp_tensor.copy_from(host_tensor);
        Shape old_shape = host_tensor.valid_shape();
        host_tensor.reshape(Shape({old_shape[0], old_shape[1], old_shape[2], old_shape[3]}));
        reorder_nchwc8_nchw(temp_tensor, host_tensor);
    }

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
        } else if (tensor.get_dtype() == AK_INT8) {
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

static std::string& replace_all(std::string&   str, const  std::string&  old_value,
                                const  std::string&  new_value) {
    while (true) {
        std::string::size_type pos(0);

        if ((pos = str.find(old_value)) != std::string::npos) {
            str.replace(pos, old_value.length(), new_value);
        } else  {
            break;
        }
    }

    return str;
}

template <typename Target_Type>
static void record_tensor_in_format(const Tensor<Target_Type>& tensor,
                                    const std::string& op_type, const std::string& op_name, bool is_out, int index) {
    std::string path = "record+" + op_type +
                       "+" + op_name +
                       "+" + (is_out ? "out" : "in") +
                       "+" + std::to_string(index) + "+";

    for (auto x : tensor.valid_shape()) {
        path += std::to_string(x) + "_";
    }

    path = replace_all(path, "/", "_");
    write_tensorfile(tensor, (path + ".txt").c_str());
}

template <typename Dtype>
static std::string vector_2_string(std::vector<Dtype> vec) {
    std::string ans = "[";

    for (auto a : vec) {
        ans += std::to_string(a) + ",";
    }

    ans += "]";
    return ans;
}

template <typename Dtype>
static void printf_intrin_var(Dtype data) {
    std::string ans = "";

    for (int i = 0; i < sizeof(data) / 4; i++) {
        ans += std::to_string(data[i]) + ",";
    }

    LOG(INFO) << ans;
}


}
}

#endif

#endif //ANAKIN_DEBUG_H
