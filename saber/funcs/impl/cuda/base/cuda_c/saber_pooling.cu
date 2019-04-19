
#include "saber/funcs/impl/cuda/saber_pooling.h"
#include "saber/funcs/impl/cuda/vender_pooling.h"
#include "saber/funcs/calibrate.h"
#include "saber/core/tensor_op.h"
#include <cfloat>

namespace anakin {
namespace saber {

template <>
SaberStatus SaberPooling<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param, Context<NV> &ctx) {
    _impl->create(inputs, outputs, param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param, Context<NV> &ctx) {

    this->_ctx = &ctx;
    _impl = new VenderPooling<NV, AK_FLOAT>;
    _impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}
template <>
SaberStatus SaberPooling<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param) {
    _impl->dispatch(inputs, outputs, param);
    return SaberSuccess;
}

union Reg{
    unsigned int idata;
    char b[4];
};

__global__ void pool_s8s8_max_c4(const int nthreads,
                                 const void* const in_data, const int channels,
                                 const int height, const int width, const int out_height,
                                 const int out_width, const int kernel_h, const int kernel_w,
                                 const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                 void* const out_data, float place_holder, float trans_scale) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % out_width;
        const int ph = (index / out_width) % out_height;
        const int c = (index / out_width / out_height) % channels;
        const int n = index / out_width / out_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        unsigned int maxval = 0x80808080; // this is magic
        const unsigned int* in_slice =
                (const unsigned int*)(in_data);
        int offset = (n * channels + c) * height * width;
        in_slice += offset;
        unsigned int *out = (unsigned int*)out_data;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                unsigned int read_in = in_slice[h * width + w];
                asm volatile (" vmax4.s32.s32.s32 %0, %1, %2, %0;"
                : "=r"(maxval) : "r"(maxval), "r"(read_in));
            }
        }

        out[index] = maxval;
    }
}
__global__ void pool_s8s8_avrg_c4(const int nthreads,
                                  const void* const in_data, const int channels,
                                  const int height, const int width, const int out_height,
                                  const int out_width, const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                  void* const out_data, float avg_1, float trans_scale) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % out_width;
        const int ph = (index / out_width) % out_height;
        const int c = (index / out_width / out_height) % channels;
        const int n = index / out_width / out_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        Reg reg;
        int sum0 = 0;
        int sum1 = 0;
        int sum2 = 0;
        int sum3 = 0;
        const unsigned int* in_slice =
                (const unsigned int*)(in_data);
        int offset = (n * channels + c) * height * width;
        in_slice += offset;
        unsigned int *out = (unsigned int*)out_data;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                reg.idata = in_slice[h * width + w];
                sum0 += reg.b[0];
                sum1 += reg.b[1];
                sum2 += reg.b[2];
                sum3 += reg.b[3];
            }
        }
        float sum0f = (float)sum0 * avg_1;
        float sum1f = (float)sum1 * avg_1;
        float sum2f = (float)sum2 * avg_1;
        float sum3f = (float)sum3 * avg_1;
        reg.b[0] = static_cast<char>(sum0f);
        reg.b[1] = static_cast<char>(sum1f);
        reg.b[2] = static_cast<char>(sum2f);
        reg.b[3] = static_cast<char>(sum3f);
//        printf("%x\n", reg.idata);
        out[index] = reg.idata;
    }
}

template <>
SaberStatus SaberPooling<NV, AK_INT8>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param, Context<NV> &ctx) {
    if (inputs[0]->get_dtype() == AK_FLOAT) {
        Shape in_shape = inputs[0]->valid_shape();
        _int8_input.re_alloc(in_shape, AK_INT8);
        _int8_input.set_scale(inputs[0]->get_scale());
        _int8_input.set_layout(Layout_NCHW_C4);
    }
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        Shape out_shape = outputs[0]->valid_shape();
        _int8_output.re_alloc(out_shape, AK_INT8);
        _int8_output.set_scale(outputs[0]->get_scale());
        _int8_output.set_layout(Layout_NCHW_C4);
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<NV, AK_INT8>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param, Context<NV> &ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberPooling<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        PoolingParam<NV> &param) {

    CHECK_GE(inputs[0]->get_scale().size(), 1) << "not found scale factor!!!";
    CHECK_GE(outputs[0]->get_scale().size(), 1) << "not found scale factor!!!";
    CHECK_EQ(inputs[0]->channel() % 4, 0) << "not a multipler of 4";

    float in_scale = inputs[0]->get_scale()[0];
    float out_scale = outputs[0]->get_scale()[0];
    int count = outputs[0]->valid_size() / 4;
    int channels = inputs[0]->channel() / 4;
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int out_height = outputs[0]->height();
    int out_width = outputs[0]->width();
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int window_h = param.window_h;
    int window_w = param.window_w;
    auto stream = _ctx->get_compute_stream();

    const void* in_data = nullptr;
    void* out_data = nullptr;

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        conv_calibrate_fp32_int8_c4(_int8_input, *inputs[0], in_scale, *(this->_ctx));
        in_data = _int8_input.data();
    } else {
        in_data = inputs[0]->data();
    }

    if (outputs[0]->get_dtype() == AK_FLOAT) {
        out_data = _int8_output.mutable_data();
    } else {
        out_data = outputs[0]->mutable_data();
    }

    float kernel_size = window_h * window_w;
    kernel_size = 1.f / kernel_size;
    switch (param.pooling_type) {
        case Pooling_max:
            pool_s8s8_max_c4 << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
                0, stream >> > (count,
                in_data, channels, height, width,
                out_height, out_width, window_h, window_w,
                stride_h, stride_w, pad_h, pad_w, out_data,
                kernel_size, in_scale / out_scale);
        break;
        case Pooling_average_include_padding:
            pool_s8s8_avrg_c4 << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
                0, stream >> > (count,
                in_data, channels, height, width,
                out_height, out_width, window_h, window_w,
                stride_h, stride_w, pad_h, pad_w, out_data,
                kernel_size, in_scale / out_scale);
        break;
        default:
            LOG(FATAL) << "not support yet!!!" << param.pooling_type;
            break;
    }
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        calibrate_int8_c4_fp32(*outputs[0], _int8_output, out_scale, *_ctx);
    }
    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, NV, AK_HALF);

}
}