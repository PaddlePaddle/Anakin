#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_8bit_pooling_kernel.h"
#include "debug.h"
namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PoolingParam<X86>& param,
        Context<X86>& ctx) {
    LayoutType in_laytype = inputs[0]->get_layout();

    if (in_laytype == Layout_NCHW) {
        return SaberSuccess;
    }

    auto src_shape = inputs[0]->shape();
    auto dst_shape = outputs[0]->shape();

    bool layout_c16 = (in_laytype == Layout_NCHW_C16R || in_laytype == Layout_NCHW_C16);

    bool layout_c8 = (in_laytype == Layout_NCHW_C8R || in_laytype == Layout_NCHW_C8);

    if (!utils::one_of(param.pooling_type,
                       Pooling_max,
                       Pooling_average_include_padding,
                       Pooling_average_exclude_padding)) {
        LOG(FATAL) << "not support " << param.pooling_type;
        return SaberUnImplError;
    }

    jit_pool_conf_t jpp;
    jpp.src_fmt = inputs[0]->get_layout();
    const int ndims = 4;
    jpp.ndims = ndims;
    jpp.mb = src_shape[0];
    jpp.c = inputs[0]->channel();

    if (in_laytype == Layout_NCHW_C8R || in_laytype == Layout_NCHW_C16R) {
        jpp.c = utils::round_up(src_shape.channel(), inputs[0]->valid_shape().get_layout_aligned_length());
    }

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
    auto pooling_type = param.pooling_type;
    //for jit always div ksize in including padding mode
    if (pooling_type == Pooling_average_include_padding && !param.pooling_padded()){
        pooling_type = Pooling_average_exclude_padding;
    }
    jpp.alg = pooling_type;
    jpp.ind_dt = AK_FLOAT;

    if (_kernel != nullptr) {
        delete _kernel;
    }

    if (layout_c16) {
        CHECK(mayiuse(avx512_common)) << "jit pooling init failed";
        CHECK(jit_pool_kernel_f32<avx512_common>::init_conf(jpp)) << "jit pooling init failed";
        _kernel = new jit_pool_kernel_f32<avx512_common>(jpp);
    } else if (layout_c8) {
        CHECK(mayiuse(avx2)) << "jit pooling init failed";
        CHECK(jit_pool_kernel_f32<avx2>::init_conf(jpp)) << "jit pooling init failed";
        _kernel = new jit_pool_kernel_f32<avx2>(jpp);
    }

    if (inputs[0]->get_dtype() != AK_FLOAT || inputs[0]->get_layout() != Layout_NCHW) {
        _input_scale.reshape(Shape({inputs[0]->num(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width()}));
        _input_scale.set_scale(inputs[0]->get_scale());
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;

    if (inputs[0]->get_dtype() != AK_FLOAT || inputs[0]->get_layout() != Layout_NCHW) {
        _input_scale.re_alloc(Shape({inputs[0]->num(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width()}),
                              AK_FLOAT);
    }

    return create(inputs, outputs, param, ctx);
}

void pooling_avx2_nchwc8(const float* src, float* dst, int in_n, int in_c, int in_h, int in_w,
                         int out_h,
                         int out_w, int stride_h, int stride_w, int window_h, int window_w, int pad_h, int pad_w,
                         PoolingType pooling_type) {
    int size_in_n = in_c * in_h * in_w * 8;
    int size_in_c = in_h * in_w * 8;
    int size_out_n = in_c * out_h * out_w * 8;
    int size_out_c = out_h * out_w * 8;

    for (int ind_n = 0; ind_n < in_n; ++ind_n) {
        for (int ind_c = 0; ind_c < in_c; ++ind_c) {
            for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                int sh = ind_h * stride_h;
                int eh = sh + window_h;

                sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
                eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;


                for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                    int sw = ind_w * stride_w;
                    int ew = sw + window_w;

                    sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
                    ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;

                    float result[8] = {0.f};

                    int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w * 8 + ind_w * 8;

                    for (int kh = sh; kh < eh; ++kh) {
                        for (int kw = sw; kw < ew; ++kw) {
                            for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                                int src_ind =
                                    ind_n * size_in_n + ind_c * size_in_c + kh * in_w * 8 + kw * 8 + inner_c_id;

                                if (kh == sh && kw == sw) {
                                    result[inner_c_id] = src[src_ind];
                                } else {
                                    if (pooling_type == Pooling_max) {
                                        result[inner_c_id] =
                                            result[inner_c_id] >= src[src_ind] ? result[inner_c_id] : src[src_ind];
                                        //                                        LOG(INFO)<<"find it "<<inner_c_id<<","<<result[inner_c_id];
                                    }

                                    if (pooling_type == Pooling_average_include_padding) {
                                        result[inner_c_id] += src[src_ind];
                                    }

                                    if (pooling_type == Pooling_average_exclude_padding) {
                                        result[inner_c_id] += src[src_ind];
                                    }
                                }

                            }
                        }
                    }

                    if (pooling_type == Pooling_average_include_padding) {

                        int bh = window_h;
                        int bw = window_w;

                        if (ew == in_w) {
                            bw = sw + window_w >= in_w + pad_w ? in_w + pad_w : sw + window_w;
                            bw -= sw;
                        }

                        if (eh == in_h) {
                            bh = sh + window_h >= in_h + pad_h ? in_h + pad_h : sh + window_h;
                            bh -= sh;
                        }

                        for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                            result[inner_c_id] /= bh * bw;
                        }
                    }

                    if (pooling_type == Pooling_average_exclude_padding) {
                        for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                            result[inner_c_id] /= (ew - sw) * (eh - sh);
                        }
                    }

                    for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {

                        dst[dst_ind + inner_c_id] = result[inner_c_id];
                        //                        LOG(INFO)<<"finnal it "<<dst_ind+inner_c_id<<","<<dst[dst_ind+inner_c_id];
                    }

                    //                    exit(0);
                    //LOG(INFO)<<"saber:"<<dst_ind<<"re:"<<result;

                }
            }
        }
    }

}

void pooling_avx2_nchwc8_nchw(const float* src, float* dst, int in_n, int in_c, int in_h, int in_w,
                              int out_h,
                              int out_w, int stride_h, int stride_w, int window_h, int window_w, int pad_h, int pad_w,
                              PoolingType pooling_type, int real_c) {
    int size_in_n = in_c * in_h * in_w * 8;
    int size_in_c = in_h * in_w * 8;
    int size_out_n = in_c * out_h * out_w * 8;
    int size_out_c = out_h * out_w * 8;
    int size_out_real_n = real_c * out_h * out_w;
    int size_out_real_c = out_h * out_w;
    #pragma omp parallel for collapse(3) schedule(static)

    for (int ind_n = 0; ind_n < in_n; ++ind_n) {
        for (int ind_c = 0; ind_c < in_c; ++ind_c) {
            for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                int sh = ind_h * stride_h;
                int eh = sh + window_h;

                sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
                eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;


                for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                    int sw = ind_w * stride_w;
                    int ew = sw + window_w;

                    sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
                    ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;

                    float result[8] = {0.f};



                    for (int kh = sh; kh < eh; ++kh) {
                        for (int kw = sw; kw < ew; ++kw) {
                            for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                                int src_ind =
                                    ind_n * size_in_n + ind_c * size_in_c + kh * in_w * 8 + kw * 8 + inner_c_id;

                                if (kh == sh && kw == sw) {
                                    result[inner_c_id] = src[src_ind];
                                } else {
                                    if (pooling_type == Pooling_max) {
                                        result[inner_c_id] =
                                            result[inner_c_id] >= src[src_ind] ? result[inner_c_id] : src[src_ind];
                                        //                                        LOG(INFO)<<"find it "<<inner_c_id<<","<<result[inner_c_id];
                                    }

                                    if (pooling_type == Pooling_average_include_padding) {
                                        result[inner_c_id] += src[src_ind];
                                    }

                                    if (pooling_type == Pooling_average_exclude_padding) {
                                        result[inner_c_id] += src[src_ind];
                                    }
                                }

                            }
                        }
                    }

                    if (pooling_type == Pooling_average_include_padding) {

                        int bh = window_h;
                        int bw = window_w;

                        if (ew == in_w) {
                            bw = sw + window_w >= in_w + pad_w ? in_w + pad_w : sw + window_w;
                            bw -= sw;
                        }

                        if (eh == in_h) {
                            bh = sh + window_h >= in_h + pad_h ? in_h + pad_h : sh + window_h;
                            bh -= sh;
                        }

                        for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                            result[inner_c_id] /= bh * bw;
                        }
                    }

                    if (pooling_type == Pooling_average_exclude_padding) {
                        for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                            result[inner_c_id] /= (ew - sw) * (eh - sh);
                        }
                    }

                    for (int inner_c_id = 0; inner_c_id < 8; inner_c_id++) {
                        int dst_ind = ind_n * size_out_real_n + (ind_c * 8 + inner_c_id) * size_out_real_c + ind_h * out_w +
                                      ind_w;
                        dst[dst_ind] = result[inner_c_id];
                        //                        LOG(INFO)<<"finnal it "<<dst_ind+inner_c_id<<","<<dst[dst_ind+inner_c_id];
                    }

                    //                    exit(0);
                    //LOG(INFO)<<"saber:"<<dst_ind<<"re:"<<result;

                }
            }
        }
    }

}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>
::dispatch(const std::vector<Tensor<X86>*>& inputs,
           std::vector<Tensor<X86>*>& outputs,
           PoolingParam<X86>& param) {

    const float* src = static_cast<const float*>(inputs[0]->data());
    float* dst = static_cast<float*>(outputs[0]->mutable_data());

    DLOG(INFO) << "input layout " << inputs[0]->get_layout() << " , output layout " <<
               outputs[0]->get_layout();

    if (_kernel != nullptr && (inputs[0]->get_layout() == Layout_NCHW_C8
                               || inputs[0]->get_layout() == Layout_NCHW_C8R) && (outputs[0]->get_layout() == Layout_NCHW_C8
                                       || outputs[0]->get_layout() == Layout_NCHW_C8R)) {

        const float* src = (const float*)inputs[0]->data();
        float* dst = (float*)outputs[0]->mutable_data();
        const auto& jpp = _kernel->jpp;
        auto ker = [&](int n, int b_c, int oh) {
            jit_pool_call_t arg;
            const int ij = oh * jpp.stride_h;
            const int i_t_overflow = std::max(0, jpp.t_pad - ij);
            const int i_b_overflow = std::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
            const int ih = std::max(ij - jpp.t_pad, 0);

            // TODO verify the calulation
            int index = n * jpp.ih * jpp.iw * jpp.c + b_c * jpp.iw * jpp.ih * jpp.c_block  + ih * jpp.iw *
                        jpp.c_block;
            arg.src = &src[index];
            index = n * jpp.oh * jpp.ow * jpp.c + b_c * jpp.ow * jpp.oh * jpp.c_block + oh * jpp.ow *
                    jpp.c_block;
            arg.dst = &dst[index];

            arg.oh = (oh == 0);
            arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
            arg.kh_padding_shift = i_t_overflow * jpp.kw;
            arg.kw_padding = 0;
            arg.ker_area_h = (float)(jpp.kh -
                                     std::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih) -
                                     std::max(0, jpp.t_pad - oh * jpp.stride_h));
            (*_kernel)(&arg);
        };

        #pragma omp parallel for collapse(3) schedule(static)

        for (int n = 0; n < jpp.mb; ++n) {
            for (int b_c = 0; b_c < jpp.nb_c; ++b_c) {
                for (int oh = 0; oh < jpp.oh; ++oh) {
                    ker(n, b_c, oh);
                }
            }
        }
    } else if (inputs[0]->get_layout() == Layout_NCHW_C8
               || inputs[0]->get_layout() == Layout_NCHW_C8R) {
        if (outputs[0]->get_layout() == Layout_NCHW_C8 || outputs[0]->get_layout() == Layout_NCHW_C8R) {
            int in_n = inputs[0]->num();
            int in_c = inputs[0]->channel() / 8;

            if (inputs[0]->get_layout() == Layout_NCHW_C8R) {
                in_c = utils::div_up(inputs[0]->channel(), 8);
                //                LOG(INFO)<<"input inputs[0]->channel()  c= "<<inputs[0]->channel()<<","<<in_c;
            }

            int in_h = inputs[0]->height();
            int in_w = inputs[0]->width();
            int out_h = outputs[0]->height();
            int out_w = outputs[0]->width();

            pooling_avx2_nchwc8(src, dst, in_n, in_c, in_h, in_w, out_h, out_w,
                                param.stride_h, param.stride_w, param.window_h, param.window_w, param.pad_h, param.pad_w,
                                param.pooling_type);
            //            write_tensorfile(*inputs[0],"input_pooling");
            //            write_tensorfile(*outputs[0],"output_pooling");
            //            exit(0);
        } else {
            //            DLOG(FATAL)<<"pooling nchw_c8 to nchw_c8r";
            int in_n = inputs[0]->num();
            int in_c = utils::div_up(inputs[0]->channel(), 8);
            int real_c = inputs[0]->channel();
            int in_h = inputs[0]->height();
            int in_w = inputs[0]->width();
            int out_h = outputs[0]->height();
            int out_w = outputs[0]->width();
            pooling_avx2_nchwc8_nchw(src, dst, in_n, in_c, in_h, in_w, out_h, out_w,
                                     param.stride_h, param.stride_w, param.window_h, param.window_w, param.pad_h, param.pad_w,
                                     param.pooling_type, real_c);
            DLOG(INFO)<<"fp32 pooling choose pooling_avx2_nchwc8_nchw";
        }
    } else {
        if (inputs[0]->get_dtype() != AK_FLOAT || inputs[0]->get_layout() != Layout_NCHW) {
            reorder_nhwc_nchw(*inputs[0], _input_scale);
            src = static_cast<const float*>(_input_scale.data());
        }

        //x86 common code
        int in_n = inputs[0]->num();
        int in_c = inputs[0]->channel();
        int in_h = inputs[0]->height();
        int in_w = inputs[0]->width();
        int size_in_n = in_c * in_h * in_w;
        int size_in_c = in_h * in_w;

        int out_h = outputs[0]->height();
        int out_w = outputs[0]->width();
        int size_out_n = in_c * out_h * out_w;
        int size_out_c = out_h * out_w;
        #pragma omp parallel for collapse(3) schedule(static)

        for (int ind_n = 0; ind_n < in_n; ++ind_n) {
            for (int ind_c = 0; ind_c < in_c; ++ind_c) {
                for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                    int sh = ind_h * param.stride_h;
                    int eh = sh + param.window_h;

                    sh = (sh - param.pad_h) < 0 ? 0 : sh - param.pad_h;
                    eh = (eh - param.pad_h) > in_h ? in_h : eh - param.pad_h;


                    for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                        int sw = ind_w * param.stride_w;
                        int ew = sw + param.window_w;

                        sw = (sw - param.pad_w) < 0 ? 0 : sw - param.pad_w;
                        ew = (ew - param.pad_w) > in_w ? in_w : ew - param.pad_w;


                        float result = static_cast<float>(0);

                        int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

                        for (int kh = sh; kh < eh; ++kh) {
                            for (int kw = sw; kw < ew; ++kw) {
                                int src_ind = ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

                                if (kh == sh && kw == sw) {
                                    result = src[src_ind];
                                } else {
                                    if (param.pooling_type == Pooling_max) {
                                        result = result >= src[src_ind] ? result : src[src_ind];
                                    }

                                    if (param.pooling_type == Pooling_average_include_padding) {
                                        result += src[src_ind];
                                    }

                                    if (param.pooling_type == Pooling_average_exclude_padding) {
                                        result += src[src_ind];
                                    }
                                }

                            }
                        }

                        if (param.pooling_type == Pooling_average_include_padding) {

                            int bh = param.window_h;
                            int bw = param.window_w;

                            if (ew == in_w) {
                                bw = sw + param.window_w >= in_w + param.pad_w ? in_w + param.pad_w : sw + param.window_w;
                                bw -= sw;
                            }

                            if (eh == in_h) {
                                bh = sh + param.window_h >= in_h + param.pad_h ? in_h + param.pad_h : sh + param.window_h;
                                bh -= sh;
                            }

                            result /= bh * bw;

                        }

                        if (param.pooling_type == Pooling_average_exclude_padding) {
                            result /= (ew - sw) * (eh - sh);
                        }

                        dst[dst_ind] = result;
                        //LOG(INFO)<<"saber:"<<dst_ind<<"re:"<<result;

                    }
                }
            }

        }
    }

    return SaberSuccess;
}


template <>
SaberStatus SaberPooling<X86, AK_INT8>::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PoolingParam<X86>& param,
        Context<X86>& ctx) {

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());

    if (!utils::one_of(param.pooling_type,
                       Pooling_max,
                       Pooling_average_include_padding,
                       Pooling_average_exclude_padding)) {
        return SaberUnImplError;
    }

    jit_pool_conf_t jpp;
    const int simd_w = 16;
    const int ndims = 4;
    jpp.src_fmt = inputs[0]->get_layout();
    jpp.ndims = ndims;
    jpp.mb = src_shape[0];
    jpp.c  = src_shape[3];
    jpp.id = (ndims == 5) ? src_shape[2] : 1;
    jpp.ih = src_shape[ndims - 3];
    jpp.iw = src_shape[ndims - 2];
    jpp.od = (ndims == 5) ? dst_shape[2] : 1;
    jpp.oh = dst_shape[ndims - 3];
    jpp.ow = dst_shape[ndims - 2];
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

    jpp.ind_dt = AK_UINT8;
    jpp.src_dt = inputs[0]->get_dtype();
    jpp.dst_dt = outputs[0]->get_dtype();

    //fixme:only support uint8 now
//    if (jpp.src_dt == AK_FLOAT) {
//        jpp.src_dt = AK_UINT8;
//    }
    CHECK_NE(jpp.src_dt , AK_FLOAT);

    if (_kernel != nullptr) {
        delete _kernel;
        _kernel = nullptr;
    }

    if (jit_avx512_core_8bit_pooling_kernel::init_conf(jpp) != SaberSuccess) {
        LOG(FATAL) << "init_conf failed";
        return SaberUnImplError;
    }

    kernel_nhwc_ = new jit_avx512_core_8bit_pooling_kernel(jpp);

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        utils::try_expand_tensor(_input_scale, inputs[0]->valid_shape());
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<X86, AK_INT8>::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PoolingParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        _input_scale.re_alloc(inputs[0]->valid_shape(), AK_UINT8);
    }

    CHECK(outputs[0]->get_layout() == Layout_NHWC);
    //FIXME:intel kernel not support scale so we pass scale from input to output
    outputs[0]->set_scale(inputs[0]->get_scale());
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberPooling<X86, AK_INT8>::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PoolingParam<X86>& param) {
    if (!mayiuse(avx512_common)) {
        LOG(FATAL) << "only run in avx512";
        return SaberUnImplError;
    }

    const auto& jpp = kernel_nhwc_->jpp;
    unsigned char* src_i8 = (unsigned char*)(inputs[0]->data());

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        utils::ScaleUtils::scale_fp32_uint8(_input_scale, *inputs[0]);
        src_i8 = static_cast<unsigned char*>(_input_scale.mutable_data());
    }

    unsigned char* dst_i8 = (unsigned char*)(outputs[0]->mutable_data());
    float* dst_f32 = (float*)(outputs[0]->mutable_data());
    auto ker = [&](int ithr, int nthr) {
        const int work_amount = jpp.mb * jpp.oh * jpp.ow;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, oh{0}, ow{0};
        nd_iterator_init(start, n, jpp.mb, oh, jpp.oh, ow, jpp.ow);

        auto p = jit_pool_call_nhwc_t();
        memset(&p, 0, sizeof(jit_pool_call_nhwc_t));

        for (int iwork = start; iwork < end; ++iwork) {
            const int ih = utils::max(oh * jpp.stride_h - jpp.t_pad, 0);
            const int iw = utils::max(ow * jpp.stride_w - jpp.l_pad, 0);
            const int kh_start = utils::max(0, jpp.t_pad - oh * jpp.stride_h);
            const int kh_end = utils::min(jpp.kh, jpp.ih + jpp.t_pad - oh * jpp.stride_h);
            const int kw_start = utils::max(0, jpp.l_pad - ow * jpp.stride_w);
            const int kw_end = utils::min(jpp.kw, jpp.iw + jpp.l_pad - ow * jpp.stride_w);
            size_t src_blk_off = n * jpp.ih * jpp.iw * jpp.c + ih * jpp.iw * jpp.c + iw * jpp.c;
            size_t dst_blk_off = n * jpp.oh * jpp.ow * jpp.c + oh * jpp.ow * jpp.c + ow * jpp.c;

            p.src_i8 = &src_i8[src_blk_off];

            if (jpp.dst_dt == AK_FLOAT) {
                p.dst_i8 = reinterpret_cast<unsigned char*>(dst_f32 + dst_blk_off);
            } else {
                p.dst_i8 = &dst_i8[dst_blk_off];
            }

            p.kw_range = (size_t)(kw_end - kw_start);
            p.kh_range = (size_t)(kh_end - kh_start);
            p.idivider = 1.0f / ((jpp.alg == Pooling_average_exclude_padding) ?
                                 p.kh_range* p.kw_range : jpp.kw * jpp.kh);

            kernel_nhwc_->ker_(&p);

            nd_iterator_step(n, jpp.mb, oh, jpp.oh, ow, jpp.ow);
        }
    };

    #pragma omp parallel
    {
        ker(anakin_get_thread_num(), anakin_get_num_threads());
    }

    return SaberSuccess;
}

template class SaberPooling<X86, AK_FLOAT>;
template class SaberPooling<X86, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, X86, AK_HALF);

}
} // namespace anakin
