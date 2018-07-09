#include "saber/lite/funcs/saber_deconv.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
namespace anakin{

namespace saber{

namespace lite{

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

SaberDeconv2D::SaberDeconv2D() {
    _workspace_fwd_sizes = 0;
    _flag_relu = false;
    _param = nullptr;
}

SaberDeconv2D::SaberDeconv2D(const ParamBase* param) {
    _param = (DeConv2DParam*)param;
    this->_flag_param = true;
}

SaberDeconv2D::~SaberDeconv2D() {}

SaberStatus SaberDeconv2D::load_param(const ParamBase *param) {
    _param = (DeConv2DParam*)(param);
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberDeconv2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    Shape output_shape = (inputs[0]->shape());

    if (!this->_flag_param) {
        printf("load deconv param first\n");
        return SaberNotInitialized;
    }

    if (inputs[0]->dims() < 4) {
        printf("using reshape2d to reshape a 1d conv?\n");
        return SaberInvalidValue;
    }

    output_shape.set_num(inputs[0]->num()); // N
    output_shape.set_channel(_param->_num_output); // K

    int kernel_extent_h = _param->_dila_h * (_param->_kh - 1) + 1;
    int output_dim_h = (inputs[0]->height() - 1) *
                       _param->_stride_h + kernel_extent_h - 2 * _param->_pad_h;
    int kernel_extent_w = _param->_dila_w * (_param->_kw - 1) + 1;
    int output_dim_w = (inputs[0]->width() - 1) *
                       _param->_stride_w + kernel_extent_w - 2 * _param->_pad_w;

    output_shape.set_height(output_dim_h);
    output_shape.set_width(output_dim_w);
    return outputs[0]->set_shape(output_shape);
}

SaberStatus SaberDeconv2D::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load deconv param first\n");
        return SaberNotInitialized;
    }

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

    // printf("kw: %d, kh: %d\n", _kw, _kh);
    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    LCHECK_EQ(chin % _param->_group, 0, "input channel or group size error");
    LCHECK_EQ(chout % _param->_group, 0, "output channel or group size error");

    //! deconv weights layout: chin * chout * kh * kw
    _m = chout * _param->_kw * _param->_kh / _param->_group;
    _n = hin * win;
    _k = chin / _param->_group;

    _workspace_data.reshape(_param->_group * _m * _n * sizeof(float));

    _gemmer.init(l1_cache, l2_cache, _m, _n, _k, true, false, threads);

    printf("Deconv: USE GEMM, numout=%d, chin=%d, kernel=%d, stride=%d, pad=%d, group=%d, win=%d, hin=%d\n", \
            chout, chin, _param->_kw, _param->_stride_w, _param->_group, win, hin);
    this->_flag_init = true;
    return SaberSuccess;
}


SaberStatus SaberDeconv2D::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init deconv first\n");
        return SaberNotInitialized;
    }

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();

    bool flag_1x1s1p1 = (_param->_kw == 1) && (_param->_kh == 1) && (_param->_stride_h == 1) && \
        (_param->_stride_w == 1) && (_param->_pad_w == 0) && (_param->_pad_h == 0) && \
        (_param->_dila_w == 1) && (_param->_dila_h == 1);


    const float* weights = _param->_weights;

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int group = _param->_group;

    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;
    int group_size_weights = chin * chout * _param->_kw * _param->_kh / (group * group);

    for (int i = 0; i < num; ++i) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        float* col_data = _workspace_data.mutable_data();
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < group; ++g) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            _gemmer(weights_group, _m, din_group, _n, coldata_group, _n, 1.f, 0.f, _flag_relu);
        }

        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, _param->_kh, _param->_kw, _param->_pad_h, _param->_pad_w, \
                _param->_stride_h, _param->_stride_w, _param->_dila_h, _param->_dila_w, dout_batch);
        }

        //! add bias
        if (_param->_bias_term) {
            fill_bias(dout_batch, _param->_bias, chout, wout * hout);
        }

    }


    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


