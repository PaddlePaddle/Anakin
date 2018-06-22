#include "saber/funcs/impl/arm/saber_deconv.h"
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"
namespace anakin{

namespace saber{

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

    this->_ctx = ctx;
    //printf("conv init \n");

    int threads = this->_ctx.get_act_ids().size();

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
    int l1_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L1_cache;
    int l2_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    if (conv_param.bias()->valid_size() > 0) {
        _bias_term = true;
    } else {
        _bias_term = false;
    }

    CHECK_EQ(chin % conv_param.group, 0) << "input channel or group size error";
    CHECK_EQ(chout % conv_param.group, 0) << "output channel or group size error";

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
    return create(inputs, outputs, conv_param, ctx);
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, ConvParam<OpTensor> &conv_param) {

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
            _gemmer(weights_group, _m, din_group, _n, coldata_group, _n, 1.f, 0.f, _flag_relu);
        }

        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, _kh, _kw, conv_param.pad_h, conv_param.pad_w, \
                conv_param.stride_h, conv_param.stride_w, conv_param.dilation_h, conv_param.dilation_w, \
                dout_batch);
        }

        //! add bias
        if (conv_param.bias()->valid_size() > 0) {
            fill_bias(dout_batch, conv_param.bias()->data(), chout, wout * hout);
        }

    }


    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


