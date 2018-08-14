//#include "saber/funcs/impl/cuda/saber_conv_act.h"
#include "saber/saber_types.h"
#include "saber/core/common.h"
namespace anakin{

namespace saber{

template <typename Dtype, bool bias_flag, bool relu_flag>
__global__ void depthwise_conv_1d(const int nthreads,
                                  const Dtype* const din, const int num, const int channels,
                                  const int hin, const int win, const int hout,
                                  const int wout, const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                  Dtype* const dout, const Dtype* const weight, const Dtype* const bias) {
    int size_channel_in = hin * win;
    int size_channel_out = hout * wout;
    int size_kernel = kernel_h * kernel_w;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % wout;
        const int ph = (index / wout) % hout;
        const int c = (index / size_channel_out) % channels;
        const int n = index / size_channel_out / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, hin + pad_h);
        int wend = min(wstart + kernel_w, win + pad_w);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hin);
        wend = min(wend, win);
        Dtype aveval = 0;
        const Dtype* const bottom_slice =
                din + (n * channels + c) * size_channel_in;
        const Dtype* const weight_slice =
                weight + c * size_kernel;

        int khstart = hend < kernel_h ? kernel_h - hend : 0;
        int kwstart = wend < kernel_w ? kernel_w - wend : 0;

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottom_slice[h * win + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
            }
        }
        if (bias_flag) {
            aveval+=bias[c];
        }
        if (relu_flag) {
            aveval = max(aveval, (Dtype)0);
        }
        dout[index] = aveval;
    }
}

template <typename Dtype, bool bias_flag, bool relu_flag>
__global__ void depthwise_conv_2d(const int channel_in_stride, const int channel_out_stride,
                                  const int kernel_size,
                                  const Dtype* const din, const int num, const int channels,
                                  const int hin, const int win, const int hout,
                                  const int wout, const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                  Dtype* const dout, const Dtype* const weight, const Dtype* const bias) {

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    //int n = blockIdx.z / channels;
    int i = blockIdx.z;
    int index = i * channel_out_stride + h * wout + w;

    if (w < wout && h < hout) {
        int hstart = h * stride_h - pad_h;
        int wstart = w * stride_w - pad_w;
        int hend = min(hstart + kernel_h, hin + pad_h);
        int wend = min(wstart + kernel_w, win + pad_w);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hin);
        wend = min(wend, win);
        Dtype aveval = 0;
        const Dtype* const bottom_slice = din + i * channel_in_stride;
        const Dtype* const weight_slice = weight + c * kernel_size;

        int khstart = hend < kernel_h? kernel_h - hend : 0;
        int kwstart = wend < kernel_w? kernel_w - wend : 0;

        for (int ih = hstart; ih < hend; ++ih) {
            for (int iw = wstart; iw < wend; ++iw) {
                aveval += bottom_slice[ih * win + iw] * weight_slice[(khstart + ih - hstart) * kernel_w + (kwstart + iw - wstart)];
            }
        }
        if (bias_flag) {
            aveval+=bias[c];
        }
        if (relu_flag) {
            aveval = max(aveval, (Dtype)0);
        }
        dout[index] = aveval;
    }
}

template <typename dtype, bool bias_flag, bool relu_flag>
SaberStatus saber_depthwise_conv_act(const dtype* input, dtype* output, \
    int num, int cin, int hin, int win, int hout, int wout, \
    int kw, int kh, int stride_w, int stride_h, \
    int pad_w, int pad_h, const dtype* weights, const dtype* bias, \
    cudaStream_t stream) {

#define D1

#ifdef D1
    const int count = num * cin * hout * wout;
#else
    dim3 block(32, 32);
    int gx = (wout + block.x - 1) / block.x;
    int gy = (hout + block.y - 1) / block.y;
    dim3 grid(gx, gy, num * cin);
    int channel_in_stride = hin * win;
    int channel_out_stride = hout * wout;
    int kernel_size = kw * kh;
#endif

    if (bias_flag) {
#ifdef D1
        depthwise_conv_1d<dtype, true, relu_flag><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
                count, input, num, cin, hin, win, hout, wout, kh, \
                kw, stride_h, stride_w, pad_h, pad_w, \
                output, weights, bias);
#else
        depthwise_conv_2d<dtype, true, relu_flag><<<grid, block, 0, stream>>>(
                channel_in_stride, channel_out_stride, kernel_size, \
                input, num, cin, hin, win, hout, wout, kh, \
                kw, stride_h, stride_w, pad_h, pad_w, \
                output, weights, bias);
#endif
    } else {
#ifdef D1
        depthwise_conv_1d<dtype, false, relu_flag><<< CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>> (
                count, input, num, cin, hin, win, hout, wout, kh, \
                kw, stride_h, stride_w, pad_h, \
                pad_w, output, weights, nullptr);
#else
        depthwise_conv_2d<dtype, false, relu_flag><<<grid, block, 0, stream>>>(
                channel_in_stride, channel_out_stride, kernel_size, \
                input, num, cin, hin, win, hout, wout, kh, \
                kw, stride_h, stride_w, pad_h, pad_w, \
                output, weights, nullptr);
#endif
    }

    return SaberSuccess;
}

#define INSTANCE_CONVACT(dtype, ifbias, ifrelu) \
template \
    SaberStatus saber_depthwise_conv_act<dtype, ifbias, ifrelu> (const dtype* input, dtype* output, \
    int num, int cin, int hin, int win, int hout, int wout, \
    int kw, int kh, int stride_w, int stride_h, \
    int pad_h, int pad_w, const dtype* weights, const dtype* bias, cudaStream_t stream);

INSTANCE_CONVACT(float, true, true);
INSTANCE_CONVACT(float, true, false);
INSTANCE_CONVACT(float, false, true);
INSTANCE_CONVACT(float, false, false);

} //namespace anakin

} //namespace anakin
