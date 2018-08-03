#include "saber/funcs/impl/arm/saber_deconv.h"
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"
namespace anakin{

namespace saber{

/**
 * \brief neon implementation to add bias and relu
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
void fill_bias_relu(float* tensor, const float* bias, int channel, int channel_size, bool flag_relu) {

    float* data = tensor;
    if(flag_relu){
        for (int j = 0; j < channel; ++j) {
            float32x4_t vbias = vdupq_n_f32(bias[j]);
            float32x4_t vzero = vdupq_n_f32(0.f);
            int i = 0;
            for (; i < channel_size - 3; i += 4) {
                float32x4_t vdata = vld1q_f32(&data[i]);
                vdata = vaddq_f32(vdata, vbias);
                float32x4_t vmax = vmaxq_f32(vdata, vzero); 
                vst1q_f32(data + i, vmax);
            }
            for (; i < channel_size; i++) {
                data[i] += bias[j];
                data[i] = data[i] > 0 ? data[i] : 0.f;
            }
            data += channel_size;
        }
    }else{
        for (int j = 0; j < channel; ++j) {
            float32x4_t vbias = vdupq_n_f32(bias[j]);
            int i = 0;
            for (; i < channel_size - 3; i += 4) {
                float32x4_t vdata = vld1q_f32(&data[i]);
                vdata = vaddq_f32(vdata, vbias);
                vst1q_f32(data + i, vdata);
            }
            for (; i < channel_size; i++) {
                data[i] += bias[j];
            } 
            data += channel_size;
       }
    }
}

/**
 * \brief basic direct deconvolution function
 */
void deconv_arm_basic(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
    const float* weights, const float* bias, \
    int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
    int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num_in = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    const int size_kernel = kernel_h * kernel_w;

    int kernel_ext_w = (kernel_w - 1) * dila_w + 1;
    int kernel_ext_h = (kernel_h - 1) * dila_h + 1;

    const int ch_out_g = ch_out / group;
    const int ch_in_g = ch_in / group;
    const int size_in_channel = w_in * h_in;
    const int size_in_batch = size_in_channel * ch_in;
    const int size_out_channel = w_out * h_out;
    const int size_out_batch = size_out_channel * ch_out;

    //printf("extend kernel size: %d, %d\n", kernel_ext_w, kernel_ext_h);
    const float *data_in = tensor_in.data();
    float *outptr = tensor_out.mutable_data();

    for (int b = 0; b < num_in; ++b) {
        float *outptr_batch = outptr + b * size_out_batch;
        const float* data_in_batch = data_in + b * size_in_batch;
#pragma omp parallel for collapse(2)
        for (int g = 0; g < group; ++g) {
            for (int c = 0; c < ch_out_g; ++c) {
                const float *inptr_group = data_in_batch + g * ch_in_g * size_in_channel;
                float *outptr_ch = outptr_batch + (g * ch_out_g + c) * size_out_channel;
                const float *weight_ch = weights + (g * ch_in_g * ch_out_g + c) * size_kernel;//conv deepwise

                float bias_value = flag_bias? bias[g * ch_out_g + c] : 0.f;
                fill_bias(outptr_ch, &bias_value, 1, w_out * h_out);

                for (int i = 0; i < h_out; ++i) {
                    for (int j = 0; j < w_out; ++j) {

                        const float *weight_ch_in = weight_ch;
/*
                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_ext_h, h_in);
                        int wend = std::min(wstart + kernel_ext_w, w_in);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);

                        int khstart = hend < kernel_ext_h? (kernel_ext_h - hend) / dila_h : 0;
                        int kwstart = wend < kernel_ext_w? (kernel_ext_w - wend) / dila_w : 0;
                       */
                       int h = i + pad_h;
                       int w = j + pad_w;
                        int hstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
                        int hend = h / stride_h + 1 < h_in ? h / stride_h + 1 : h_in;
                        int wstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
                        int wend = w / stride_w + 1 < w_in ? w / stride_w + 1 : w_in;

                        int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
                        int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

                        //printf("channel: %d, index: %d, %d, %d, %d, %d, %d\n", c, hstart, wstart, hend, wend, khstart, kwstart);
                        const float* inptr_ch = inptr_group + hstart * w_in + wstart;

                        for (int k = 0; k < ch_in_g; ++k) {
                            const float *weight_ch_in1 = weight_ch_in + k * ch_out_g * size_kernel;
                            const float* inptr_kernel = inptr_ch;
                            //int khidx = khstart;
                            for (int idxh = hstart; idxh < hend; idxh += dila_h) {
                                const float* inptr_kernel_w = inptr_kernel;
                               // int kwidx = kwstart;
                                for (int idxw = wstart; idxw < wend; idxw += dila_w) {
                                    int khidx = khstart - (idxh - hstart) * stride_h;
                                    int kwidx = kwstart - (idxw - wstart) * stride_w;
                                    outptr_ch[j] += weight_ch_in1[khidx * kernel_w + kwidx] * inptr_kernel_w[0];
                                    inptr_kernel_w += dila_w;
                                }
                                inptr_kernel += dila_h * w_in;
                            }
                            inptr_ch += size_in_channel;
                           // weight_ch_in += size_kernel;
                        }
                        if (flag_relu) {
                            outptr_ch[j] = outptr_ch[j] > 0? outptr_ch[j] : 0.f;
                        }
                    }
                    outptr_ch += w_out;
                }
            }
        }
    }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void col2im(const Dtype* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype* data_im) {
    memset(data_im, 0, height * width * channels * sizeof(Dtype));
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        data_col += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

template <>
SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberDeconv2D() {
    _workspace_fwd_sizes = 0;
    _flag_relu = false;
    _bias_term = true;
    _workspace_data = std::make_shared<Buffer<ARM>>();
}

template <>
SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberDeconv2D() {
     //LOG(ERROR) << "release saber conv: kw=" << _kw << ", kh=" << _kh << ", num_out=" << _conv_param.weight()->num() << \
        ", chin=" << _conv_param.weight()->channel();
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs,\
    ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) {

    //LOG(INFO) << "conv create";

    this->_ctx = &ctx;
    //printf("conv init \n");

    int threads = 1;
    this->_ctx->get_mode(threads);

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    _kw = conv_param.weight()->width();
    _kh = conv_param.weight()->height();
   // printf("kw: %d, kh: %d\n", _kw, _kh);
    int l1_cache = this->_ctx->devs[this->_ctx->get_device_id()]._info._L1_cache;
    int l2_cache = this->_ctx->devs[this->_ctx->get_device_id()]._info._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    if (conv_param.bias()->valid_size() > 0) {
        _bias_term = true;
    } else {
        _bias_term = false;
    }

//LOG(INFO) << "chin:" << chin << ", chout:" << chout << ", g: " <<conv_param.group;
    if (chin != chout || conv_param.group != chin) {
        CHECK_EQ(chin % conv_param.group, 0) << "input channel or group size error";
        CHECK_EQ(chout % conv_param.group, 0) << "output channel or group size error";
    }
    

    //! deconv weights layout: chin * chout * kh * kw
    _m = chout * _kw * _kh / conv_param.group;
    _n = hin * win;
    _k = chin / conv_param.group;

    _workspace_data->re_alloc(conv_param.group * _m * _n * sizeof(float));

    _gemmer.init(l1_cache, l2_cache, _m, _n, _k, true, false, threads);

    LOG(ERROR) << "USE GEMM";
    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) {
    //LOG(INFO) << " conv create";
    return create(inputs, outputs, conv_param, ctx);
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, ConvParam<OpTensor> &conv_param) {

    // LOG(INFO) << "Deconv start" ;

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();

    bool flag_1x1s1p1 = (_kw == 1) && (_kh == 1) && (conv_param.stride_h == 1) && \
        (conv_param.stride_w == 1) && (conv_param.pad_w == 1) && (conv_param.pad_h == 1) && \
        (conv_param.dilation_w == 1) && (conv_param.dilation_h == 1);


    const float* weights = conv_param.weight()->data();

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int group = conv_param.group;

    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;
    int group_size_weights = chin * chout * _kw * _kh / (group * group);

    for (int i = 0; i < num; ++i) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        float* col_data = (float*)_workspace_data->get_data_mutable();
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < conv_param.group; ++g) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            /*for(int j = 0; j < 8; j++){
                LOG(INFO) << "i: " << j << "weights: " << weights_group[j];
                LOG(INFO) << "i: " << j << "data: " << din_group[j];
            }*/
            //LOG(INFO) << "Deconv start" ;
            if (conv_param.bias()->valid_size() == 0) {
                _gemmer(weights_group, _m, din_group, _n, coldata_group, _n, 1.f, 0.f, _flag_relu);
            }else{
                _gemmer(weights_group, _m, din_group, _n, coldata_group, _n, 1.f, 0.f, false);
            }
            //_gemmer(weights_group, _m, din_group, _n, coldata_group, _n, 1.f, 0.f, _flag_relu);
        }
        //LOG(INFO) << "Deconv mid" ;
        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, _kh, _kw, conv_param.pad_h, conv_param.pad_w, \
                conv_param.stride_h, conv_param.stride_w, conv_param.dilation_h, conv_param.dilation_w, \
                dout_batch);
        }

        //! add bias
        if (conv_param.bias()->valid_size() > 0) {
           // fill_bias(dout_batch, conv_param.bias()->data(), chout, wout * hout);
            fill_bias_relu(dout_batch, conv_param.bias()->data(), chout, wout * hout, _flag_relu);
           // LOG(INFO) << "Deconv end" ;
        }

    }


    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


