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

#ifndef USE_SGX
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/core/tensor.h"
#include "saber/funcs/saber_util.h"
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
template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value;
    return os.str();
}
template <typename HostType>
static void reorder_nhwc_nchw(const Tensor<HostType>& input,
                              Tensor<HostType>& output) {




    int n_value = input.num();
    int c_value = input.channel();
    int h_value = input.height();
    int w_value = input.width();

    if (input.get_layout() == Layout_NHWC && output.get_layout() == Layout_NCHW) {
        if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_FLOAT) {
            float* output_ptr = static_cast<float*>(output.mutable_data());
            CHECK(input.get_scale().size() >= 1);
            float scale = input.get_scale()[0];
            const int8_t* input_ptr = static_cast<const int8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index] * scale;
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_FLOAT) {
            LOG(INFO) << "print uint 8";
            CHECK(input.get_scale().size() >= 1);
            float scale = (input.get_scale()[0]) * (127.f / 255.f);
            LOG(INFO) << "scale = " << scale;
            double sum = 0.0;
            double max = 0.0;
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());
            float* output_ptr = static_cast<float*>(output.mutable_data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = (float)input_ptr[in_index] * scale;
                            sum += output_ptr[out_index];
                            max = output_ptr[out_index] > max ? output_ptr[out_index] : max;
                        }
                    }
                }
            }

            LOG(INFO) << "avg = " << (sum / input.valid_size()) << "," << max;
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_UINT8) {
            LOG(INFO) << "reorder uint 8";
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
            const float* input_ptr = static_cast<const float*>(input.data());
            float* output_ptr = static_cast<float*>(output.mutable_data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else {
            LOG(FATAL) << "not support input type " << input.get_dtype();
        }
    } else if (input.get_layout() == Layout_NCHW && output.get_layout() == Layout_NHWC) {
        if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
            float* output_ptr = static_cast<float*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_UINT8) {
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_INT8) {
            int8_t* output_ptr = static_cast<int8_t*>(output.mutable_data());
            const int8_t* input_ptr = static_cast<const int8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_INT8) {
            CHECK(output.get_scale().size() >= 1);
            float scale = 1.f / (output.get_scale()[0]);
            int8_t* output_ptr = static_cast<int8_t*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = saturate<int8_t>(roundf(input_ptr[in_index] * scale));
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_UINT8) {
            CHECK(output.get_scale().size() >= 1);
            float scale = 1.f / (output.get_scale()[0]* (127.f / 255.f));
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = saturate<uint8_t>(roundf(input_ptr[in_index] * scale));
                        }
                    }
                }
            }
        }else {
            LOG(FATAL) << "not support in/ou type " << input.get_dtype() << "," << output.get_dtype();
        }
    } else {
        LOG(FATAL) << "not support layout " << input.get_layout() << "," << output.get_layout();
    }

}

template <typename HostType>
static void reorder_nchwc_nchw(Tensor<HostType>& input,
                               Tensor<HostType>& output) {
    if (input.valid_shape() == output.valid_shape()) {
        output.copy_from(input);
        return;
    }

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    LayoutType in_layout = input.get_layout();
    LayoutType out_layout = output.get_layout();
    bool is_nchwc_nchw = (in_layout == Layout_NCHW_C16R || in_layout == Layout_NCHW_C8R)
                         && (out_layout == Layout_NCHW);
    bool is_nchw_nchwc = (out_layout == Layout_NCHW_C16R || out_layout == Layout_NCHW_C8R)
                         && (in_layout == Layout_NCHW);
    CHECK(is_nchw_nchwc || is_nchwc_nchw) << "not support " << input.get_layout();

    if (is_nchwc_nchw) {
        Shape shape = output.valid_shape();
        int n_value = shape[0];
        int c_value = shape[1];
        int h_value = shape[2];
        int w_value = shape[3];
        Shape shape_input = input.valid_shape();
        int aligned_length = shape_input.get_layout_aligned_length();
        CHECK_GT(aligned_length, 0) << "input aligned should > 0";
        int c_round_divk = shape_input[1];

        c_round_divk = (shape_input.channel() + aligned_length - 1) / aligned_length;

        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());
        #pragma omp parallel for collapse(4) schedule(static)

        for (int n = 0; n < n_value; ++n) {
            for (int c = 0; c < c_value; ++c) {
                for (int h = 0; h < h_value; ++h) {
                    //#pragma ivdep
                    for (int w = 0; w < w_value; ++w) {
                        int round_c = c / aligned_length;
                        int remainder_c = c % aligned_length;
                        int input_idx = n * c_round_divk * h_value * w_value * aligned_length + round_c * h_value *
                                        w_value * aligned_length +
                                        h * w_value * aligned_length + w * aligned_length + remainder_c;
                        int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                         h * w_value  + w ;

                        *(output_ptr + output_idx) = input_ptr[input_idx];
                    }
                }
            }
        }
    } else if (is_nchw_nchwc) {
        Shape shape = input.valid_shape();
        int n_value = shape[0], c_value = shape[1], h_value = shape[2], w_value = shape[3];

        int aligned_length = output.valid_shape().get_layout_aligned_length();
        CHECK_GT(aligned_length, 0) << "input aligned should > 0";

        int c_round_divk = (c_value + aligned_length - 1) / aligned_length;

        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());
        #pragma omp parallel for collapse(5) schedule(static)

        for (int n = 0; n < n_value; ++n) {
            for (int c_idx = 0; c_idx < c_round_divk; ++c_idx) {
                for (int h = 0; h < h_value; ++h) {
                    for (int w = 0; w < w_value; ++w) {
                        for (int c = 0; c < aligned_length; ++c) {
                            int input_idx = n * c_value * h_value * w_value + (c_idx * aligned_length + c) * h_value * w_value +
                                            h * w_value + w;
                            int output_idx = n * c_round_divk * h_value * w_value * aligned_length + c_idx * h_value * w_value *
                                             aligned_length +
                                             h * w_value * aligned_length + w * aligned_length + c;

                            *(output_ptr + output_idx) = ((c_idx * aligned_length + c) < c_value) ? *
                                                         (input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }

    } else {
        LOG(FATAL) << "not support this shape";
    }


}

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

template <typename HOST_TYPE>
inline void calibrate_int8c4_to_fp32_host(Tensor<HOST_TYPE>& host_tensor,
        const Tensor <HOST_TYPE>& int8_tensor) {

    CHECK_EQ(host_tensor.get_dtype(), AK_FLOAT);
    CHECK_EQ(host_tensor.get_layout(), Layout_NCHW);
    CHECK_EQ(int8_tensor.get_dtype(), AK_INT8);
    CHECK_EQ(int8_tensor.get_layout(), Layout_NCHW_C4);
    CHECK_EQ(host_tensor.valid_size(), int8_tensor.valid_size());
    CHECK_GE(int8_tensor.get_scale().size(), 1);

    Shape out_stride = host_tensor.get_stride();
    Shape in_shape = int8_tensor.valid_shape();
    Shape out_shape = host_tensor.valid_shape();
    int valid_width = in_shape.width();
    int valid_height = in_shape.height();
    int valid_channel_4 = in_shape.channel() / 4;
    int valid_num = in_shape.num();
    int in_n_stride = in_shape[1] * in_shape[2] * in_shape[3] / 4;
    int in_c_stride = in_shape[2] * in_shape[3];
    int in_h_stride = in_shape[3];
    int in_w_stride = 1;

    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] / 4;
    const char* in_data = (const char*)int8_tensor.data();
    float* out_data = (float*)host_tensor.mutable_data();
    float scale = int8_tensor.get_scale()[0];

    for (int gid = 0; gid < count; ++ gid) {
        float load0, load1, load2, load3;

        int read_w = (gid) % valid_width;
        int read_h = (gid / (in_h_stride)) % valid_height;
        int read_c = (gid / (in_c_stride)) % valid_channel_4;
        int read_n = (gid / (in_n_stride)) % valid_num;

        int in_offset = read_n * in_n_stride
                        + read_c * in_c_stride
                        + read_h * in_h_stride
                        + read_w;

        int out_offset = read_n * out_stride[0]
                         + read_c * (out_stride[1] << 2)
                         + read_h * out_stride[2]
                         + read_w * out_stride[3];

        if (gid < count) {

            char readin0 = in_data[4 * in_offset + 0];
            char readin1 = in_data[4 * in_offset + 1];
            char readin2 = in_data[4 * in_offset + 2];
            char readin3 = in_data[4 * in_offset + 3];

            load0 = static_cast<float>(readin0);
            load1 = static_cast<float>(readin1);
            load2 = static_cast<float>(readin2);
            load3 = static_cast<float>(readin3);

            out_data[out_offset] = load0 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load1 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load2 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load3 * scale;
        }
    }
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

template <typename Target_Type>
static void record_tensor_in_io_format(const Tensor<Target_Type>& tensor, std::string tensor_name,
                                       bool is_out, int index, int iter = 0) {
    CHECK_EQ(tensor.get_dtype(), AK_FLOAT) << "now record func only support ak_float";
    CHECK_EQ(tensor.get_layout(), Layout_NCHW) << "now record func only support ak_float";
    std::string path = "";
    path = path + "record+" + (is_out ? "out+" : "in+") + tensor_name + "+";

    for (auto x : tensor.valid_shape()) {
        path += to_string(x) + "_";
    }

    path += "+nchw+";
    path += "ak_float+";
    path += to_string(iter);

    path = replace_all(path, "/", "_");
    write_tensorfile(tensor, (path + ".txt").c_str());
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
