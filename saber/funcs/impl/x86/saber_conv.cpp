
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_im2col_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_group_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_dwconv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv1x1.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_1x1_conv.h"
#include "saber/funcs/impl/x86/gemm_x8s8s32x_conv.h"
#include "saber/funcs/impl/x86/saber_conv_1x1.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_dwconv.h"
#include "saber/funcs/impl/x86/winograd.h"
#include "saber/funcs/debug.h"
namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    if (_input_trans) {
        int in = inputs[0]->num();
        int ic = inputs[0]->channel();
        int ih = inputs[0]->height();
        int iw = inputs[0]->width();
        utils::try_expand_tensor(_input_trans_tensor, Shape({in, ic, ih, iw},
                                 _input_trans_tensor.get_layout()));
        _input_trans_tensor.set_seq_offset(inputs[0]->get_seq_offset());
    }

    if (_input_trans) {
        return this->impl->create(_fake_input_vec, outputs, conv_elt_param, ctx);
    } else {
        return this->impl->create(inputs, outputs, conv_elt_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    bool use_avx512 = mayiuse(avx512_common);
    bool use_avx2 = mayiuse(avx2);
    int group = param.group;
    int oc = outputs[0]->channel();
    int ic = inputs[0]->channel();
    int kh = param.weight()->height();
    int kw = param.weight()->width();
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int dilation_h = param.dilation_h;
    int dilation_w = param.dilation_w;
    int ih = inputs[0]->height();
    int iw = inputs[0]->width();
    int in = inputs[0]->num();
    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType out_layout = outputs[0]->get_layout();

    this->impl = nullptr;

    bool conv_1x1_flag = (kh == 1 && kw == 1) && (pad_h == 0 && pad_w == 0) && (stride_h == 1
                         && stride_w == 1) && group == 1;
    bool is_c16 = (input_layout == Layout_NCHW_C16R) && (out_layout == Layout_NCHW_C16R) ;
    bool is_strict_c16 = is_c16 && (ic % 16 == 0 && oc % 16 == 0);
    bool is_first_c16 = (input_layout == Layout_NCHW) && (ic == 1 || ic == 3)
                        && (out_layout == Layout_NCHW_C16R || out_layout == Layout_NHWC);
    bool is_c8 = (input_layout == Layout_NCHW_C8R) && (out_layout == Layout_NCHW_C8R);
    bool is_strict_c8 = is_c8 && (ic % 8 == 0 && oc % 8 == 0);
    bool is_c8_in = (input_layout == Layout_NCHW_C8R);
    bool is_strict_c8_in = is_c8_in && (ic % 8 == 0 && oc % 8 == 0);
    bool is_c8_out = (out_layout == Layout_NCHW_C8R);
    bool is_strict_c8_out = is_c8_out && (ic % 8 == 0 && oc % 8 == 0);

    bool is_winorgrad = (kh == 3 && kw == 3) && (stride_h == 1 && stride_w == 1) && (dilation_h == 1
                        && dilation_w == 1) && group == 1;
#ifndef USE_SGX
    if (is_winorgrad && (oc >= 16 && ic >= 16 && ih >= 12 && iw >= 12)
            && (((input_layout == Layout_NCHW) && (out_layout == Layout_NCHW)))) {
        this->impl = new SaberConvWinograd<AK_FLOAT>;
    } else
#endif
    if (conv_1x1_flag && (input_layout == Layout_NCHW) && (out_layout == Layout_NCHW)) {
        this->impl = new SaberConv1X1<AK_FLOAT>;
    } else if ((use_avx2 || use_avx512) && (oc == group && ic == group) && (is_strict_c8_out
               || is_strict_c16)) {
        if (is_strict_c8_out && input_layout != Layout_NCHW_C8R) {
            _input_trans = true;
            _input_trans_tensor.re_alloc(Shape({in, ic, ih, iw}, Layout_NCHW_C8R));
            _input_trans_tensor.set_seq_offset(inputs[0]->get_seq_offset());
        }

        this->impl = new JitUniDWConv<AK_FLOAT>;
    } else if (use_avx512  && conv_1x1_flag && is_strict_c16) {
        this->impl = new JitAvx512Conv1x1<AK_FLOAT>;
    } else if (use_avx512 && param.group == 1 && (is_strict_c16 || is_first_c16)) {
        this->impl = new JitAvx512Conv<AK_FLOAT>;
    } else if (use_avx2 && param.group == 1 && pad_w <=3) {
        this->impl = new JitAvx2Conv<AK_FLOAT>;
    } else if (use_avx2 && param.group != 1 && is_strict_c8 && pad_w <=3) {
        this->impl = new JitAvx2GroupConv<AK_FLOAT>;
    }

    _fake_input_vec.push_back(&_input_trans_tensor);

    const std::vector<Tensor<X86> *> &maybe_trans = _input_trans ? _fake_input_vec : inputs;

    // sometimes the kernels above cannot be successfully intialized. In those cases,
    // test if we can fall back to SaberIm2colConv
    if (!(this->impl && this->impl->init(maybe_trans, outputs, conv_elt_param, ctx) == SaberSuccess)) {
        if (input_layout == Layout_NCHW && out_layout == Layout_NCHW) {
            this->impl = new SaberIm2colConv<AK_FLOAT>;
        } else {
            LOG(FATAL) << "not support conv for in shape = " << inputs[0]->valid_shape() << ", out shape "
                       << outputs[0]->valid_shape() << ", group = " << group;
        }
        return this->impl->init(maybe_trans, outputs, conv_elt_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    if (_input_trans) {
        _input_trans_tensor.set_seq_offset(inputs[0]->get_seq_offset());
        input_reorder_nChwc8(*inputs[0], _input_trans_tensor);
        return this->impl->dispatch(_fake_input_vec, outputs, conv_elt_param);
    } else {
        return this->impl->dispatch(inputs, outputs, conv_elt_param);
    }

    return SaberSuccess;
}


template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
create(const std::vector<Tensor<X86> *>& inputs,
       std::vector<Tensor<X86> *>& outputs,
       ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    if (inputs[0]->get_dtype() == AK_FLOAT || inputs[0]->get_layout() != Layout_NHWC) {
        _input_scale.reshape(
                Shape({inputs[0]->num(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel()},
                      Layout_NHWC));
    }

    if (outputs[0]->get_layout() == Layout_NCHW) {
        _output_scale.reshape(Shape({outputs[0]->num(), outputs[0]->height(), outputs[0]->width(),
                                      outputs[0]->channel()}, Layout_NHWC));
    }

    if (_input_vec.size() == 0 && _output_vec.size() == 0) {
        return this->impl->create(inputs, outputs, conv_elt_param, ctx);
    } else if (_input_vec.size() >= 0 && _output_vec.size() == 0) {
        return this->impl->create(_input_vec, outputs, conv_elt_param, ctx);
    } else if (_input_vec.size() == 0 && _output_vec.size() >= 0) {
        return this->impl->create(inputs, _output_vec, conv_elt_param, ctx);
    } else if (_input_vec.size() >= 0 && _output_vec.size() >= 0) {
        return this->impl->create(_input_vec, _output_vec, conv_elt_param, ctx);
    } else {
        LOG(FATAL) << "this is bug";
        return SaberUnImplError;
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
init(const std::vector<Tensor<X86> *>& inputs,
     std::vector<Tensor<X86> *>& outputs,
     ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    ConvParam<X86>* conv_param = &(param);
    CHECK(inputs[0]->get_layout() == Layout_NHWC || inputs[0]->get_dtype() == AK_FLOAT);
    const int group = param.group;
    const int oc = outputs[0]->channel();
    const int ic = inputs[0]->channel();
    const int kh = param.weight()->height();
    const int kw = param.weight()->width();
    const int pad_h = param.pad_h;
    const int pad_w = param.pad_w;
    const int stride_h = param.stride_h;
    const int stride_w = param.stride_w;
    const int dilation_h = param.dilation_h;
    const int dilation_w = param.dilation_w;
    const DataType in_dtyp = inputs[0]->get_dtype();
    bool conv_1x1_flag = (kh == 1 && kw == 1) && (pad_h == 0 && pad_w == 0) && (stride_h == 1
                         && stride_w == 1) && group == 1;

    if (inputs[0]->get_dtype() == AK_FLOAT || inputs[0]->get_layout() != Layout_NHWC) {
        if (inputs[0]->get_dtype() == AK_FLOAT) {
            if (0) { //(inputs[0]->get_posstive_flag()){
                _input_scale.re_alloc(
                    Shape({inputs[0]->num(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel()},
                          Layout_NHWC), AK_UINT8);
            } else {
                _input_scale.re_alloc(
                    Shape({inputs[0]->num(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel()},
                          Layout_NHWC), AK_INT8);
            }
        } else {
            _input_scale.re_alloc(
                Shape({inputs[0]->num(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel()},
                      Layout_NHWC), inputs[0]->get_dtype());
        }

        _input_scale.set_scale(inputs[0]->get_scale());
        _input_vec.push_back(&_input_scale);
    }

    if (outputs[0]->get_layout() == Layout_NCHW) {
        _output_scale.re_alloc(Shape({outputs[0]->num(), outputs[0]->height(), outputs[0]->width(),
                                      outputs[0]->channel()}, Layout_NHWC), outputs[0]->get_dtype());
        _output_scale.set_scale(outputs[0]->get_scale());
        _output_vec.push_back(&_output_scale);
    } else if (outputs[0]->get_layout() != Layout_NHWC) {
        LOG(FATAL) << "not support output layout " << outputs[0]->get_layout();
    }

    DLOG(INFO) << "init int8 conv group = " << group << ",pad = " << pad_h << "," << pad_w <<
               ", stride = " << stride_h << "," << stride_w
               << ", dili = " << dilation_h << "," << dilation_w << " , oc = " << oc << ", ic = " << ic <<
               ", kh = " << kh << ",kw = " << kw;
#if 0

    this->impl = new GemmX8S8S32XConv();
#else
    bool is_dw = (group > 1 && group == oc && group == ic);

    if (conv_1x1_flag && in_dtyp == AK_UINT8) {
        this->impl = new JitAvx512x8s8s32xConv1x1();
    } else if ((is_dw || group == 1) && pad_w <= 14) {
        this->impl = new JitAvx512X8S8S32XConv();
    } else {
        this->impl = new GemmX8S8S32XConv();
    }

#endif



    if (_input_vec.size() == 0 && _output_vec.size() == 0) {
        return this->impl->init(inputs, outputs, conv_elt_param, ctx);
    } else if (_input_vec.size() >= 0 && _output_vec.size() == 0) {
        return this->impl->init(_input_vec, outputs, conv_elt_param, ctx);
    } else if (_input_vec.size() == 0 && _output_vec.size() >= 0) {
        return this->impl->init(inputs, _output_vec, conv_elt_param, ctx);
    } else if (_input_vec.size() >= 0 && _output_vec.size() >= 0) {
        return this->impl->init(_input_vec, _output_vec, conv_elt_param, ctx);
    } else {
        LOG(FATAL) << "this is bug";
        return SaberUnImplError;
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    if (_input_vec.size() > 0) {
        DLOG(INFO) << "trans input dtype " << _input_vec[0]->get_dtype() << "," <<
                   _input_vec[0]->get_layout() << "," << _input_vec[0]->get_scale()[0];
    }

    if (_input_vec.size() == 0 && _output_vec.size() == 0) {
        return this->impl->dispatch(inputs, outputs, conv_elt_param);
    } else if (_input_vec.size() >= 0 && _output_vec.size() == 0) {
        reorder_nhwc_nchw(*inputs[0], _input_scale);
        return this->impl->dispatch(_input_vec, outputs, conv_elt_param);
    } else if (_input_vec.size() == 0 && _output_vec.size() >= 0) {
        auto status = this->impl->dispatch(inputs, _output_vec, conv_elt_param);
        reorder_nhwc_nchw(_output_scale, *outputs[0]);
        return  status;
    } else if (_input_vec.size() >= 0 && _output_vec.size() >= 0) {
        reorder_nhwc_nchw(*inputs[0], _input_scale);
        auto status = this->impl->dispatch(_input_vec, _output_vec, conv_elt_param);
        reorder_nhwc_nchw(_output_scale, *outputs[0]);
        return  status;
    } else {
        LOG(FATAL) << "this is bug";
    }

    return SaberSuccess;
}



DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, X86, AK_HALF);
}
}
