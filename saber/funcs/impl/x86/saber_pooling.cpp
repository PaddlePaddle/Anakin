#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"


namespace anakin {
namespace saber {

using namespace jit;
template <typename HTensor>
static void pooling_basic(HTensor& tensor_out, \
                          HTensor& tensor_in, PoolingType type, bool global, \
                          int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h) {
    //no need to pad input tensor, border is zero pad inside this function
    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int size_channel_in = w_in * h_in;

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    int size_channel_out = w_out * h_out;

    float* data_out = tensor_out.mutable_data();
    const float* data_in = tensor_in.data();

    if (global) {
        switch (type) {
        case Pooling_max:
            for (int n = 0; n < num; ++n) {
                float* data_out_batch = data_out + n * ch_out * size_channel_out;
                const float* data_in_batch = data_in + n * ch_in * size_channel_in;

                for (int c = 0; c < ch_out; ++c) {
                    const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                    data_out_batch[c] = data_in_channel[0];

                    for (int i = 0; i < size_channel_in; ++i) {
                        data_out_batch[c] = data_out_batch[c] > data_in_channel[i] ? \
                                            data_out_batch[c] : data_in_channel[i];
                    }
                }
            }

            break;

        case Pooling_average_include_padding:

        case Pooling_average_exclude_padding:
            for (int n = 0; n < num; ++n) {
                float* data_out_batch = data_out + n * ch_out * size_channel_out;
                const float* data_in_batch = data_in + n * ch_in * size_channel_in;

                for (int c = 0; c < ch_out; ++c) {
                    const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                    float sum = 0.f;

                    for (int i = 0; i < size_channel_in; ++i) {
                        sum += data_in_channel[i];
                    }

                    data_out_batch[c] = sum / size_channel_in;
                }
            }

            break;

        default:
            LOG(INFO) << "not support";
        }

        return;
    }

    switch (type) {
    case Pooling_max:
        for (int n = 0; n < num; ++n) {
            float* data_out_channel = data_out + n * ch_out * size_channel_out;
            const float* data_in_batch = data_in + n * ch_in * size_channel_in;

            for (int q = 0; q < ch_out; q++) {

                float* data_out_row = data_out_channel + q * size_channel_out;
                const float* data_in_channel = data_in_batch + q * size_channel_in;

                for (int i = 0; i < h_out; i++) {
                    for (int j = 0; j < w_out; j++) {
                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_h, h_in + pad_h);
                        int wend = std::min(wstart + kernel_w, w_in + pad_w);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);
                        hend = std::min(hend, h_in);
                        wend = std::min(wend, w_in);

                        data_out_row[j] = data_in_channel[hstart * w_in + wstart];

                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                data_out_row[j] = data_out_row[j] > \
                                                  data_in_channel[h * w_in + w] ? \
                                                  data_out_row[j] : data_in_channel[h * w_in + w];
                            }
                        }
                    }

                    data_out_row += w_out;
                }
            }
        }

        break;

    case Pooling_average_include_padding:
        for (int n = 0; n < num; ++n) {
            int pool_size = kernel_w * kernel_h;//(hend - hstart) * (wend - wstart);//problem
            float* data_out_channel = data_out + n * ch_out * size_channel_out;
            const float* data_in_batch = data_in + n * ch_in * size_channel_in;

            for (int q = 0; q < ch_out; q++) {

                float* data_out_row = data_out_channel + q * size_channel_out;
                const float* data_in_channel = data_in_batch + q * size_channel_in;

                for (int i = 0; i < h_out; i++) {
                    for (int j = 0; j < w_out; j++) {
                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_h, h_in + pad_h);
                        int wend = std::min(wstart + kernel_w, w_in + pad_w);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);
                        hend = std::min(hend, h_in);
                        wend = std::min(wend, w_in);

                        data_out_row[j] = data_in_channel[hstart * w_in + wstart];
                        float sum = 0.f;

                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                sum += data_in_channel[h * w_in + w];
                            }
                        }

                        data_out_row[j] = sum / pool_size;
                    }

                    data_out_row += w_out;
                }
            }
        }

        break;

    case Pooling_average_exclude_padding:
        for (int n = 0; n < num; ++n) {
            float* data_out_channel = data_out + n * ch_out * size_channel_out;
            const float* data_in_batch = data_in + n * ch_in * size_channel_in;

            for (int q = 0; q < ch_out; q++) {

                float* data_out_row = data_out_channel + q * size_channel_out;
                const float* data_in_channel = data_in_batch + q * size_channel_in;

                for (int i = 0; i < h_out; i++) {
                    for (int j = 0; j < w_out; j++) {
                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_h, h_in + pad_h);
                        int wend = std::min(wstart + kernel_w, w_in + pad_w);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);
                        hend = std::min(hend, h_in);
                        wend = std::min(wend, w_in);

                        data_out_row[j] = data_in_channel[hstart * w_in + wstart];
                        float sum = 0.f;

                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                sum += data_in_channel[h * w_in + w];
                            }
                        }

                        int pool_size = (hend - hstart) * (wend - wstart);
                        data_out_row[j] = sum / pool_size;
                    }

                    data_out_row += w_out;
                }
            }
        }

        break;

    default:
        LOG(FATAL) << "not support";
    }
}


template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::init(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
PoolingParam<OpTensor>& param, Context<X86>& ctx) {

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::create(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                PoolingParam<OpTensor>& param,
Context<X86>& ctx) {
    if(std::is_same<LayOutType_in,NCHW>::value&&std::is_same<LayOutType_out,NCHW>::value&&std::is_same<LayOutType_op,NCHW>::value){
        return SaberSuccess;
    }
    jit_pool_conf_t jpp_;

    if (init_conf(jpp_, inputs, outputs, param) != SaberSuccess) {
        return SaberUnImplError;
    }

    kernel_ = new jit_uni_pool_kernel_f32<avx512_common>(jpp_);
    return SaberSuccess;
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>
            ::dispatch(const std::vector<DataTensor_in*>& inputs,
                       std::vector<DataTensor_out*>& outputs,
PoolingParam<OpTensor>& param) {
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    if(std::is_same<LayOutType_in,NCHW>::value&&std::is_same<LayOutType_out,NCHW>::value&&std::is_same<LayOutType_op,NCHW>::value){

        pooling_basic(*outputs[0],*inputs[0],param.pooling_type,param.global_pooling,param.window_w,param.window_h,param.stride_w,param.stride_h,param.pad_w,param.pad_h);
        return SaberSuccess;
    }

    return SaberSuccess;

}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::init_conf(
                jit_pool_conf_t& jpp, const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
PoolingParam<OpTensor>& param) {

    using namespace utils;

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    bool ok = true
              && mayiuse(avx512_common)
              && std::is_same<LayOutType_in, NCHW_C16>::value
              && std::is_same<LayOutType_op, NCHW>::value
              && one_of(param.pooling_type, Pooling_max,
                        Pooling_average_include_padding,
                        Pooling_average_exclude_padding);

    if (!ok) {
        return SaberUnImplError;
    }

    const int simd_w = 16;
    const int ndims = 4;

    jpp.ndims = ndims;
    jpp.mb = src_shape[0];
    jpp.c = src_shape[1] * 16;
    jpp.id = (ndims == 5) ? src_shape[2] : 1;
    jpp.ih = src_shape[ndims - 2];
    jpp.iw = src_shape[ndims - 1];
    jpp.od = (ndims == 5) ? dst_shape[2] : 1;
    jpp.oh = dst_shape[ndims - 2];
    jpp.ow = dst_shape[ndims - 1];

    jpp.stride_d = 1;
    jpp.stride_h = param.stride_h;
    jpp.stride_w = param.stride_w;
    jpp.kd = 1;
    jpp.kh = param.window_h;
    jpp.kw = param.window_w;

    jpp.f_pad = 0;
    jpp.t_pad = param.pad_h;
    jpp.l_pad = param.pad_w;

    jpp.alg = param.pooling_type;

    jpp.ind_dt = AK_FLOAT;

    jpp.simple_alg = false;

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;

    if (jpp.alg == Pooling_max) {
        jpp.ur_w = 16;
    } else {
        jpp.ur_w = 24;
    }

    if (jpp.ow < jpp.ur_w) {
        jpp.ur_w = jpp.ow;
    }

    if (jpp.l_pad > jpp.ur_w) {
        return SaberUnImplError;
    }

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    if (jit_uni_pool_kernel_f32<avx512_common>::init_conf(jpp)) {
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}
template class SaberPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class SaberPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
} // namespace anakin
