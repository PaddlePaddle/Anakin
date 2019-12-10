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
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <immintrin.h>

#ifndef USE_SGX
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/core/tensor.h"
#include "saber/funcs/saber_util.h"
namespace anakin {
namespace saber {


template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value;
    return os.str();
}
template <typename Target_Type>
static void write_tensorfile(const Tensor<Target_Type>& tensor, const char* locate,
                             bool trans_tensor = true) {

    typedef typename DefaultHostType<Target_Type>::Host_type HOST_TYPE;
    Tensor<HOST_TYPE> host_tensor;

    if (trans_tensor) {
        if (tensor.get_dtype() == AK_INT8 && tensor.get_layout() == Layout_NCHW_C4) {
            Tensor<HOST_TYPE> temp_tensor;
            temp_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
            temp_tensor.copy_from(tensor);
            temp_tensor.set_scale(tensor.get_scale());
            Shape fp32_shape = tensor.valid_shape();
            fp32_shape.set_layout(Layout_NCHW);
            host_tensor.re_alloc(fp32_shape, AK_FLOAT);
            calibrate_int8c4_to_fp32_host(host_tensor, temp_tensor);
        } else if (tensor.get_layout() == Layout_NHWC) {
            Tensor<HOST_TYPE> temp_tensor;
            temp_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
            temp_tensor.copy_from(tensor);
            LOG(INFO) << "scale size = " << tensor.get_scale().size();
            LOG(INFO) << "scale value = " << tensor.get_scale()[0];
            temp_tensor.set_scale(tensor.get_scale());
            Shape fp32_shape = tensor.valid_shape();
            fp32_shape.set_layout(Layout_NCHW);
            host_tensor.re_alloc(fp32_shape, AK_FLOAT);
            reorder_nhwc_nchw(temp_tensor, host_tensor);
            LOG(INFO) << "record int8 tensor";
            //        calibrate_int8nhwc_to_fp32_host(host_tensor, temp_tensor);
        } else {
            host_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
            host_tensor.copy_from(tensor);
        }

        if (host_tensor.get_layout() == Layout_NCHW_C8R) {
            Tensor<HOST_TYPE> temp_tensor(host_tensor.valid_shape());
            temp_tensor.copy_from(host_tensor);
            Shape old_shape = host_tensor.valid_shape();
            host_tensor.reshape(Shape({old_shape[0], old_shape[1], old_shape[2], old_shape[3]}));
            reorder_nchwc8_nchw(temp_tensor, host_tensor);
        }
    } else {
        host_tensor.re_alloc(tensor.valid_shape(), tensor.get_dtype());
        host_tensor.copy_from(tensor);
    }

    LOG(INFO) << "target tensor data:" << tensor.valid_size();
    FILE* fp = fopen(locate, "w");

    if (fp == nullptr) {
        LOG(ERROR) << "file open field " << locate;
    } else {
        if (host_tensor.get_dtype() == AK_FLOAT) {
            const float* data_ptr = (const float*)host_tensor.data();
            int size = host_tensor.valid_size();

            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %f \n", i, (data_ptr[i]));
            }
        } else if (host_tensor.get_dtype() == AK_INT8) {
            const char* data_ptr = (const char*)host_tensor.data();
            int size = host_tensor.valid_size();

            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %d \n", i, (data_ptr[i]));
            }
        } else if (host_tensor.get_dtype() == AK_UINT8) {
            const unsigned char* data_ptr = (const unsigned char*)host_tensor.data();
            int size = host_tensor.valid_size();

            for (int i = 0; i < size; ++i) {
                fprintf(fp, "[%d] %u \n", i, (data_ptr[i]));
            }
        } else {
            LOG(FATAL) << "not supported write type";
        }

        if (tensor.get_seq_offset().size() > 0) {
            auto seq_offset = tensor.get_seq_offset();

            for (int i = 0; i < seq_offset.size(); i++) {
                for (int offset_data : seq_offset[i]) {
                    fprintf(fp, "[offset_%d] %d \n", i, offset_data);
                }
            }
        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}

static void split_string(const std::string& s, char delim,
                         std::vector<std::string>& elems) {
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}


static std::string replace_all(std::string   str, const  std::string&  old_value,
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
                                    const std::string& op_type, const std::string& op_name,
                                    bool is_out, int index, int iter = 0) {
    //    CHECK_EQ(tensor.get_dtype(), AK_FLOAT) << "now record func only support ak_float";
    std::string path = "record+" + op_type +
                       "+" + op_name +
                       "+" + (is_out ? "out" : "in") +
                       "+" + to_string(index) + "+";

    if (tensor.valid_size() > 1 && tensor.shape().size() == 4) {
        path += to_string(tensor.num()) + "_" + to_string(tensor.channel()) + "_" +
                to_string(tensor.height()) + "_" + to_string(tensor.width()) + "_";
    } else {
        for (auto x : tensor.valid_shape()) {
            path += to_string(x) + "_";
        }
    }

    path += "+nchw+";
    path += "ak_float+";
    path += to_string(iter);

    path = replace_all(path, "/", "_");
    write_tensorfile(tensor, (path + ".txt").c_str());
}
static void get_shape(std::string shape_string, std::vector<int>& shape_vec) {
    std::vector<std::string> shape_s_vec;
    split_string(shape_string, '_', shape_s_vec);
    shape_vec.clear();

    for (int i = 0; i < shape_s_vec.size(); i++) {
        shape_vec.push_back(atoi(shape_s_vec[i].c_str()));
    }
}
static std::string get_basename(std::string path) {
    std::vector<std::string> elems;
    split_string(path, '/', elems);

    if (elems.size() >= 1) {
        return elems[elems.size() - 1];
    } else {
        return "";
    }
}

template <typename Target_Type>
static void read_tensor(Tensor<Target_Type>& tensor, std::string location) {
    FILE* fp = fopen(location.c_str(), "r");
    float* tensor_data = static_cast<float*>(tensor.mutable_data());
    int index = 0;

    if (fp == nullptr) {
        LOG(FATAL) << "can`t open " << location;
    } else {
        char buf[1024];
        std::vector<int> seq_offset;

        while (fgets(buf, 1024, fp) != NULL) {
            std::string str(buf);
            std::vector<std::string> s_vec;
            split_string(str, ' ', s_vec);

            if (s_vec[0].find("offset") != std::string::npos) {
                if (s_vec[0] == "[offset_0]") {
                    seq_offset.push_back(atoi(s_vec[1].c_str()));
                } else {
                    LOG(FATAL) << "not support " << s_vec[0];
                }
            } else {
                CHECK_LT(index, tensor.valid_size()) << "index must less than valid size";
                tensor_data[index++] = atof(s_vec[1].c_str());
            }
        }
    }

}

template <typename Target_Type>
static void load_tensor_in_io_format(Tensor<Target_Type>& tensor, bool& is_input,
                                     std::string& op_name, std::string location) {
    std::string base_name(get_basename(location));
    LOG(INFO) << "base name " << base_name;
    std::vector<std::string> base_split;
    split_string(base_name, '+', base_split);
    op_name = base_split[2];
    std::string in_out_flag = base_split[3];
    std::string shape = base_split[5];
    std::string layout = base_split[6];
    std::string data_type = base_split[7];
    std::vector<int> shape_vec;
    get_shape(shape, shape_vec);
    CHECK(in_out_flag == "in"
          || in_out_flag == "out") << "in/out flag must be in or out, not " << in_out_flag;
    CHECK(layout == "nchw") << "load layout now only support nchw not " << layout;
    CHECK(data_type == "ak_float") << "data type now only support ak_float not " << data_type;
    is_input = in_out_flag == "in";
    Shape ak_shape(shape_vec, Layout_NCHW);
    tensor.re_alloc(ak_shape);
    read_tensor(tensor, location);
}


template <typename Dtype>
static std::string vector_2_string(std::vector<Dtype> vec) {
    std::string ans = "[";

    for (auto a : vec) {
        ans += to_string(a) + ",";
    }

    ans += "]";
    return ans;
}

template <typename Dtype>
static void printf_intrin_var(Dtype data) {
    std::string ans = "";

    for (int i = 0; i < sizeof(data) / 4; i++) {
        ans += to_string(data[i]) + ",";
    }

    LOG(INFO) << ans;
}

template <typename Dtype>
static void printf_intrin_var_epi16(Dtype data) {
    std::string ans = "";

    for (int i = 0; i < sizeof(data) / 4; i++) {
        ans += to_string(data[i]) + ",";
    }

    LOG(INFO) << ans;
}

template <typename Dtype>
static void printf_pointer(Dtype* data, size_t length) {
    std::string ans = "";

    for (int i = 0; i < length; i++) {
        ans += to_string(data[i]) + ",";
    }

    LOG(INFO) << ans << " [length = "<<length<<"] \n";
}
template <>
void printf_pointer<uint8_t >(uint8_t* data, size_t length){
    std::string ans = "";

    for (int i = 0; i < length; i++) {
        ans += to_string((int)data[i]) + ",";
    }

    LOG(INFO) << ans << " [length = "<<length<<"] \n";
}

template <>
void printf_pointer<int8_t >(int8_t* data, size_t length){
    std::string ans = "";

    for (int i = 0; i < length; i++) {
        ans += to_string((int)data[i]) + ",";
    }

    LOG(INFO) << ans << " [length = "<<length<<"] \n";
}
template <>
void printf_pointer<void>(void* data, size_t length){
    LOG(INFO)<<"printf_pointer do not want to print void*";
}

#if defined(__AVX2__)

template<>
void printf_intrin_var<__m256i>(__m256i data) {
    int avx2_print_buf[8];
    std::string ans = "";
    _mm256_storeu_si256((__m256i*)(&avx2_print_buf[0]), data);

    for (int i = 0; i < 8; i++) {
        ans += to_string(avx2_print_buf[i]) + ",";
    }

    LOG(INFO) << ans;
}
template<>
void printf_intrin_var<__m256>(__m256 data) {
    float avx2_print_buf[8];
    std::string ans = "";
    _mm256_storeu_ps((&avx2_print_buf[0]), data);

    for (int i = 0; i < 8; i++) {
        ans += to_string(avx2_print_buf[i]) + ",";
    }

    LOG(INFO) << ans;
}
template<>
void printf_intrin_var_epi16<__m256i>(__m256i data) {
    short avx2_print_buf[16];
    std::string ans = "";
    _mm256_storeu_si256((__m256i*)(&avx2_print_buf[0]), data);

    for (int i = 0; i < 16; i++) {
        ans += to_string(avx2_print_buf[i]) + ",";
    }

    std::cout << ans << std::endl;
}
#endif

#if defined(__AVX512F__)
template<>
void printf_intrin_var<__m512i>(__m512i data) {
    std::string ans = "";
    int avx512_print_buf[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    _mm512_storeu_si512((__m512i*)(&avx512_print_buf[0]), data);

    for (int i = 0; i < 16; i++) {
        ans += to_string(avx512_print_buf[i]) + ",";
    }

    LOG(INFO) << ans;
}
template<>
void printf_intrin_var<__v32hi>(__v32hi data) {
    std::string ans = "";
    short avx512_print_buf[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                                  - 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                                 };
    _mm512_storeu_si512((__m512i*)(&avx512_print_buf[0]), (__m512i)data);

    for (int i = 0; i < 32; i++) {
        ans += to_string(avx512_print_buf[i]) + ",";
    }
    LOG(INFO) << ans;
}
#endif

}
}

#endif

#endif //ANAKIN_DEBUG_H
