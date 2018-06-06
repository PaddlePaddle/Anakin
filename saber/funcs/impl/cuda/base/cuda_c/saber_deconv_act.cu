#include "saber/funcs/impl/cuda/saber_deconv_act.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

namespace anakin{

namespace saber{

template <typename dtype, bool flag_bias, bool flag_act>
__global__ void direct_deconv(const dtype* const din,
                              const dtype* bias_data, const dtype* const weight_data,
                              const int num, const int in_channels, const int out_channels,
                              const int hout,const int wout, const int channel_out_stride,
                              const int hin, const int win, const int channel_in_stride,
                              const int kernel_h, const int kernel_w, const int kernel_size,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w,
                              const int dilation_h, const int dilation_w,
                              dtype* dout) {

    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int w =  wo + pad_w;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int h =  ho + pad_h;
    int iout = blockIdx.z;
    int cout = iout % out_channels;
    int n = iout / out_channels;
    int iin = n * in_channels;
    int idx_out = iout * channel_out_stride + ho * wout + wo;

    extern __shared__ dtype sharedw[];

    dtype val = 0;

    if (wo < wout && ho < hout) {
        for(int ic = 0; ic < in_channels; ic++) {
            //! read weights
            int idx_weight = threadIdx.y * blockDim.x + threadIdx.x;
            if (idx_weight < kernel_size) {
                sharedw[idx_weight] = weight_data[(cout * in_channels + ic) * kernel_size + idx_weight];
            }
            __syncthreads();
            //! get start and end index
            const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            const int phend = min(h / stride_h + 1, hin);
            const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
            const int pwend = min(w / stride_w + 1, win);

            const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
            const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

            const dtype* const din_c = din + (iin + ic) * channel_in_stride;

            //! start computation
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int kh = khstart - (ph - phstart) * stride_h;
                    int kw = kwstart - (pw - pwstart) * stride_w;
                    val += din_c[ph * win + pw] * sharedw[kh * kernel_w + kw];
                }
            }
        }
        //! finnal computation
        if (flag_bias) {
            val += bias_data[cout];
        }
        if (flag_act) {
            val = val > (dtype)0? val : (dtype)0;
        }
        dout[idx_out] = val;

    }
}

template <typename dtype, bool bias_flag, bool relu_flag>
__global__ void depthwise_deconv_2d(const int channel_in_stride, const int channel_out_stride,
                                    const int kernel_size,
                                    const dtype* const din, const int num, const int channels,
                                    const int hin, const int win, const int hout,
                                    const int wout, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                    dtype* const dout, const dtype* const weight, const dtype* const bias) {

    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int w =  wo + pad_w;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int h =  ho + pad_h;
    int c = blockIdx.z % channels;
    //int n = blockIdx.z / channels;
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

        const int khstart=(h >= kernel_h) ? ((h-kernel_h)%stride_h)+(kernel_h-stride_h): h;
        const int kwstart=(w >= kernel_w) ? ((w-kernel_w)%stride_w)+(kernel_w-stride_w) : w;

        dtype gradient = 0;
        const dtype* const top_diff_slice = din + i * channel_in_stride;

        const dtype* const weight_slice = weight + c * kernel_size;

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int kh = khstart - (ph - phstart) * stride_h;
                int kw = kwstart - (pw - pwstart) * stride_w;
                gradient += top_diff_slice[ph * win + pw] * sharedw[kh * kernel_w + kw];
                //gradient += top_diff_slice[ph * win + pw] * weight_slice[kh * kernel_w + kw];
            }
        }
        if (bias_flag) {
            gradient += bias[c];
        }
        if (relu_flag) {
            gradient = gradient > (dtype)0? gradient : (dtype)0;
        }
        dout[index] = gradient;
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberDeconv2DAct<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs,
    ConvActiveParam<OpTensor>& param) {
    cudaStream_t stream = this->_ctx.get_compute_stream();
    //! inputs only has one tensor

    const InDataType* din = inputs[0]->data();
    OutDataType* dout = outputs[0]->mutable_data();
    const OpDataType* weight = param.conv_param.weight()->data();

    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int num = inputs[0]->num();
    int ch_in = inputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int ch_out = outputs[0]->channel();

    int kernel_w = param.conv_param.weight()->width();
    int kernel_h = param.conv_param.weight()->height();

    dim3 block(32, 32);
    int gx = (wout + block.x - 1) / block.x;
    int gy = (hout + block.y - 1) / block.y;
    dim3 grid(gx, gy, num * ch_out);
    int channel_in_stride = hin * win;
    int channel_out_stride = hout * wout;
    int kernel_size = kernel_h * kernel_w;
    int shared_mem_size = kernel_size * sizeof(OpDataType);

    if (_use_k4_s2_p1) {
        const InDataType * bias_data = (param.conv_param.bias()->valid_size() > 0) ?
                                  param.conv_param.bias()->data() : NULL;
        const OpDataType *weights_data = new_weights_dev.data();
        ker_deconv_implicit_gemm_k4_s2_p1_32x32_relu(dout, din,
                                                weights_data, bias_data,
                                                num,
                                                hin, win, hout, wout,
                                                ch_in, ch_out, stream);
        return SaberSuccess;
    }

    if (param.conv_param.bias()->valid_size() > 0) { // deconv with bias
        const InDataType* bias = param.conv_param.bias()->data();
        //! depthwise deconv
        if (param.conv_param.group == ch_in && ch_in == ch_out) {
            depthwise_deconv_2d<InDataType, true, true><<<grid, block, shared_mem_size, stream>>>(
                    channel_in_stride, channel_out_stride, kernel_size, \
                din, num, ch_in, hin, win, hout, wout, kernel_h, \
                kernel_w, param.conv_param.stride_h, param.conv_param.stride_w, \
                param.conv_param.pad_h, param.conv_param.pad_w, \
                dout, weight, bias);
        } else {
            direct_deconv<InDataType, true, true><<<grid, block,  shared_mem_size, stream>>>
            (din, bias, weight,
                num, ch_in, ch_out, hout, wout, channel_out_stride,
                hin, win, channel_in_stride,
                kernel_h, kernel_w, kernel_size,
                param.conv_param.stride_h, param.conv_param.stride_w,
                param.conv_param.pad_h, param.conv_param.pad_w,
                param.conv_param.dilation_h, param.conv_param.dilation_w,
                dout);

        }
    } else { //deconv without bias
        //! depthwise deconv
        if (param.conv_param.group == ch_in && ch_in == ch_out) {
            depthwise_deconv_2d<InDataType, false, true> << < grid, block, shared_mem_size, stream>> > (
                    channel_in_stride, channel_out_stride, kernel_size, \
                din, num, ch_in, hin, win, hout, wout, kernel_h, \
                kernel_w, param.conv_param.stride_h, param.conv_param.stride_w, \
                param.conv_param.pad_h, param.conv_param.pad_w, \
                dout, weight, nullptr);
        } else {
            direct_deconv<InDataType, true, true><<<grid, block,  shared_mem_size, stream>>>
                    (din, nullptr, weight, num, ch_in, ch_out, hout, wout, channel_out_stride,
                            hin, win, channel_in_stride, kernel_h, kernel_w, kernel_size,
                            param.conv_param.stride_h, param.conv_param.stride_w,
                            param.conv_param.pad_h, param.conv_param.pad_w,
                            param.conv_param.dilation_h, param.conv_param.dilation_w,
                            dout);
        }
    }

    return SaberSuccess;
}

} //namespace anakin

} //namespace anakin
