#ifndef ANAKIN_SABER_FUNCS_CALIBRATE_H
#define ANAKIN_SABER_FUNCS_CALIBRATE_H

#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include <vector>
namespace anakin {
namespace saber {

template<typename TargetType>
SaberStatus conv_calibrate_fp32_int8_c4(
        Tensor<TargetType> &out_tensor,
        const Tensor<TargetType> &in_tensor,
        const float in_scale, Context<TargetType> ctx);

template<typename TargetType>
SaberStatus conv_calibrate_int32_fp32(
        Tensor<TargetType> &out_tensor,
        const Tensor<TargetType> &in_tensor,
        const float in_scale, const float* weight_scale,
        Context<TargetType> ctx);

template<typename TargetType>
SaberStatus conv_calibrate_int8_c4_fp32(
        Tensor<TargetType> &out_tensor,
        const Tensor<TargetType> &in_tensor,
        const float* weight_scale,
        Context<TargetType> ctx);

template<typename TargetType>
void float2char(bool col_direct, signed char* dst, const float* src,
                float *scale, int height, int width,
                Context<TargetType> ctx);

template<typename TargetType>
void fix2float(float * dst,
               const float *sA, const float *sB,
               const float alpha, const float beta, int height, int width,
               Context<TargetType> ctx);

template <typename TargetType>
SaberStatus get_tensor_scale(std::vector<float> &vector_scale,
        const Tensor<TargetType> &tensor, const int axis) {

    int out_dims = tensor.valid_shape()[axis];
    vector_scale.resize(out_dims);
    long long inner_dim = tensor.count_valid(axis + 1, tensor.dims());

    const float* in_data = (const float*)(tensor.data());

    for (int c = 0; c < out_dims; ++c) {
        float max_val = -1.f;

        for (int i = 0; i < inner_dim; ++i) {
            float read_data = fabs(in_data[i]);
            max_val = (read_data > max_val) ? read_data : max_val;
        }

        vector_scale[c] = max_val / 127.f;
        in_data += inner_dim;
    }
}

template<typename TargetType, typename TargetType_H>
SaberStatus convert_weights_to_nchw_c4_host(Tensor<TargetType_H>& out_tensor,
                                            const Tensor<TargetType_H>& in_tensor,
                                            Context<TargetType> ctx) {

    int input_channel = in_tensor.channel();
    int output_channel = out_tensor.num();
    std::vector<float> vector_weight_scale;
    get_tensor_scale(vector_weight_scale, in_tensor, 0);

    int o_num = out_tensor.num();
    int o_channel = out_tensor.valid_shape()[1];
    int o_height = out_tensor.height();
    int o_width = out_tensor.width();

    int out_n_stride = o_channel * o_height * o_width;
    int out_c_stride = o_height * o_width;
    int out_h_stride = o_width;

    Shape in_stride = in_tensor.get_stride();
    const float* in_weight_data = (const float*)in_tensor.data();
    char* out_weight_data = (char*)out_tensor.mutable_data();

    for (int idx = 0; idx < o_num * o_channel * o_height * o_width; ++idx) {

        int n = (idx / (out_n_stride)) % o_num;
        int in_offset = ((idx / (out_n_stride)) % o_num) * in_stride[0]
                        + ((idx / (out_c_stride)) % o_channel) * (in_stride[1] * 4)
                        + ((idx / (out_h_stride)) % o_height) * in_stride[2]
                        + (idx % o_width) * in_stride[3];

        int out_offset = ((idx / (out_n_stride)) % o_num) * out_n_stride
                         + ((idx / (out_c_stride)) % o_channel) * out_c_stride
                         + ((idx / (out_h_stride)) % o_height) * out_h_stride
                         + (idx % o_width);

        out_weight_data[out_offset * 4 + 0] = (char)(round(
                in_weight_data[in_offset + 0 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 1] = (char)(round(
                in_weight_data[in_offset + 1 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 2] = (char)(round(
                in_weight_data[in_offset + 2 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 3] = (char)(round(
                in_weight_data[in_offset + 3 * in_stride[1]] / vector_weight_scale[n]));
    }
    out_tensor.set_scale(vector_weight_scale);
//    for (auto i : vector_weight_scale) {
//        LOG(INFO) << i;
//    }
    return SaberSuccess;
}
template<typename TargetType, typename TargetType_H>
SaberStatus convert_bias_host(Tensor<TargetType_H>& out_tensor,
                              const Tensor<TargetType_H>& in_tensor,
                              float in_scale, std::vector<float> vector_weight_scale,
                              Context<TargetType> ctx) {
    unsigned long weight_size = vector_weight_scale.size();
    unsigned long bias_size = in_tensor.size();
            CHECK_GT(in_scale, 0);
            CHECK_GT(weight_size, 0);
            CHECK_EQ(bias_size, weight_size);

    const float* in_data = (const float*)in_tensor.data();
    float* out_data = (float*)out_tensor.mutable_data();

    for (int i = 0; i < bias_size; ++i) {
        out_data[i] = in_data[i] / in_scale / vector_weight_scale[i];
    }

    return SaberSuccess;
}

} // namespace saber
} // namespace anakin

#endif
