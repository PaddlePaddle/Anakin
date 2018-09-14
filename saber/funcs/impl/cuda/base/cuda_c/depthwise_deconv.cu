#include "saber/funcs/impl/cuda/depthwise_deconv.h"
#include "saber/funcs/impl/impl_macro.h"
namespace anakin {
namespace saber {

template <typename dtype>
__global__ void depthwise_deconv_2d(const int channel_in_stride, const int channel_out_stride,
                                    const int kernel_size,
                                    const dtype* const din, const int num, const int channels,
                                    const int hin, const int win, const int hout,
                                    const int wout, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                    dtype* const dout, const dtype* const weight, const dtype* const bias,
                                    bool bias_flag, bool relu_flag) {

    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int w =  wo + pad_w;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int h =  ho + pad_h;
    int c = blockIdx.z % channels;
    int i = blockIdx.z;
    int index = i * channel_out_stride + ho * wout + wo;

    extern __shared__ dtype sharedw[];
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (idx < kernel_size) {
        sharedw[idx] = weight[c * kernel_size + idx];
    }
    __syncthreads();

    if (wo < wout && ho < hout) {
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, hin);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, win);

        const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
        const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

        dtype gradient = 0;
        const dtype* const top_diff_slice = din + i * channel_in_stride;

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int kh = khstart - (ph - phstart) * stride_h;
                int kw = kwstart - (pw - pwstart) * stride_w;
                gradient += top_diff_slice[ph * win + pw] * sharedw[kh * kernel_w + kw];
            }
        }
        if (bias_flag) {
            gradient += bias[c];
        }

        if (relu_flag) {
            gradient = gradient > (dtype)0 ? gradient : (dtype)0;
        }
        dout[index] = gradient;
    }
}

template <>
SaberStatus DepthwiseDeconv<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    return SaberSuccess;
}

template <>
SaberStatus DepthwiseDeconv<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus DepthwiseDeconv<NV, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();
    const float* weight = (const float*)param.weight()->data();
    const float* bias = (const float*)param.bias()->data();

    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int num = inputs[0]->num();
    int ch_in = inputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int ch_out = outputs[0]->channel();

    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();

    dim3 block(32, 32);
    int gx = (wout + block.x - 1) / block.x;
    int gy = (hout + block.y - 1) / block.y;
    dim3 grid(gx, gy, num * ch_out);
    int channel_in_stride = hin * win;
    int channel_out_stride = hout * wout;
    int kernel_size = kernel_h * kernel_w;
    int shared_mem_size = kernel_size * sizeof(float);

    bool bias_flag = param.bias()->valid_size() > 0;
    bool relu_flag = param.activation_param.has_active;

    depthwise_deconv_2d<float><<<grid, block, shared_mem_size, stream>>>(
            channel_in_stride, channel_out_stride, kernel_size, \
                din, num, ch_in, hin, win, hout, wout, kernel_h, \
                kernel_w, param.stride_h, param.stride_w, \
                param.pad_h, param.pad_w, \
                dout, weight, bias, bias_flag, relu_flag);

    return SaberSuccess;
}

template class DepthwiseDeconv<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(DepthwiseDeconv, ConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(DepthwiseDeconv, ConvParam, NV, AK_INT8);

} //namespace anakin

} //namespace anakin
