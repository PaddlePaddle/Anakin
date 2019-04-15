#include "anakin_config.h"
#ifndef USE_SGX
#include "saber/funcs/impl/x86/vender_deconv.h"
#include "saber/funcs/impl/x86/mkldnn_helper.h"

namespace anakin {
namespace saber {

template <DataType Dtype>
SaberStatus VenderDeconv2D<X86, Dtype>::init_conv_prv(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs, ConvParam<X86>& param) {

    _engine = std::make_shared<mkldnn::engine>(mkldnn::engine::cpu, 0);
    _alg = mkldnn::algorithm::deconvolution_direct;
    _stream = std::make_shared<mkldnn::stream>(mkldnn::stream::kind::eager);

    Shape in_sh = inputs[0]->valid_shape();
    Shape out_sh = outputs[0]->valid_shape();
    std::vector<int> b_sh = {out_sh.channel()};
    std::vector<int> w_sh = param.weight()->valid_shape();

    auto in_md = create_mkldnn_memory_desc<Dtype>(in_sh);
    auto bias_md = create_mkldnn_memory_desc<Dtype>(b_sh);
    auto weights_md = create_mkldnn_memory_desc<Dtype>(w_sh);
    auto out_md = create_mkldnn_memory_desc<Dtype>(out_sh);

    mkldnn_mem_dim strides = {param.stride_h, param.stride_w};
    mkldnn_mem_dim dilation = {param.dilation_h, param.dilation_w};
    mkldnn_mem_dim padding = {param.pad_h, param.pad_w};

    bool with_bias = param.bias() && param.bias() -> valid_size() > 0 ? true : false;
    bool with_dilation = (param.dilation_w == 1 && param.dilation_h == 1) ? false : true;

    //TODO:here we ignored group
    std::shared_ptr<desc<mkldnn_deconv> > conv_desc;

    if (with_bias && with_dilation) {
        conv_desc = std::make_shared<desc<mkldnn_deconv> >(mkldnn::prop_kind::forward_inference, _alg,
                    in_md, weights_md, bias_md, out_md, strides, dilation, padding, padding,
                    mkldnn::padding_kind::zero);
    } else if (with_bias) {
        conv_desc = std::make_shared<desc<mkldnn_deconv> >(mkldnn::prop_kind::forward_inference, _alg,
                    in_md, weights_md, bias_md, out_md, strides, padding, padding,
                    mkldnn::padding_kind::zero);
    } else if (with_dilation) {
        conv_desc = std::make_shared<desc<mkldnn_deconv> >(mkldnn::prop_kind::forward_inference, _alg,
                    in_md, weights_md, out_md, strides, dilation, padding, padding,
                    mkldnn::padding_kind::zero);
    } else {
        conv_desc = std::make_shared<desc<mkldnn_deconv> >(mkldnn::prop_kind::forward_inference, _alg,
                    in_md, weights_md, out_md, strides, padding, padding,
                    mkldnn::padding_kind::zero);
        LOG(INFO)<<"it is me";
    }

    pdesc<mkldnn_deconv> conv_prv_desc = pdesc<mkldnn_deconv>(*conv_desc, *_engine);
    //above: make convolution_primitive_description
    //below: make memorys

    //make input_memory and weights_memory for user
    _in_mem = create_mkldnn_memory_no_data(inputs[0], *_engine);
    _w_mem = create_mkldnn_memory(param.mutable_weight(), w_sh,
                                  mkldnn_mem_format::oihw, mkldnn_mem_dtype::f32, *_engine);

    //set input_memory and weights_memory for conv
    _conv_in_mem = _in_mem;

    if (pdesc<mkldnn_mem>(conv_prv_desc.src_primitive_desc()) != _in_mem->get_primitive_desc()) {
        _conv_in_mem.reset(new mkldnn_mem(conv_prv_desc.src_primitive_desc()));
        _prvs.push_back(mkldnn::reorder(*_in_mem, *_conv_in_mem));
    }

    //std::vector<mkldnn::primitive> weights_trans;
    _conv_w_mem = _w_mem;
//    LOG(INFO)<<"conv weight mem "<<conv_prv_desc.weights_primitive_desc().desc().data.format;
//    LOG(INFO)<<"weight mem "<<_w_mem->get_primitive_desc().desc().data.format;
//    if (pdesc<mkldnn_mem>(conv_prv_desc.weights_primitive_desc()) != _w_mem->get_primitive_desc()) {
//        _conv_w_mem.reset(new mkldnn_mem(conv_prv_desc.weights_primitive_desc()));
//        //weights_trans.push_back(mkldnn::reorder(w_mem, conv_w_mem));
//        _prvs_weights_trans.push_back(mkldnn::reorder(*_w_mem, *_conv_w_mem));
//        mkldnn::stream(mkldnn::stream::kind::eager).submit(_prvs_weights_trans).wait();
//
//        LOG(INFO)<<"change weights";
//    }

    //set output_memory for user and conv
    _out_mem = create_mkldnn_memory_no_data(outputs[0], *_engine);
    _conv_out_mem = _out_mem;

    if (pdesc<mkldnn_mem>(conv_prv_desc.dst_primitive_desc()) != _out_mem->get_primitive_desc()) {
        _conv_out_mem.reset(new mkldnn_mem(conv_prv_desc.dst_primitive_desc()));
    }

    //set bias_memory for user and conv
    //make convolution primitive
    if (with_bias) {
        _bias_mem = create_mkldnn_memory(param.mutable_bias(), b_sh,
                                         mkldnn_mem_format::x, mkldnn_mem_dtype::f32, *_engine);
        _conv_bias_mem = _bias_mem;

        if (pdesc<mkldnn_mem>(conv_prv_desc.bias_primitive_desc()) != _bias_mem->get_primitive_desc()) {
            _conv_bias_mem.reset(new mkldnn_mem(conv_prv_desc.bias_primitive_desc()));
            _prvs_weights_trans.push_back(mkldnn::reorder(*_bias_mem, *_conv_bias_mem));
        }

        _prvs.push_back(mkldnn_deconv(conv_prv_desc, *_conv_in_mem, *_conv_w_mem, *_conv_bias_mem,
                                    *_conv_out_mem));
    } else {
        LOG(INFO)<<"no bias";
        _prvs.push_back(mkldnn_deconv(conv_prv_desc, *_conv_in_mem, *_conv_w_mem, *_conv_out_mem));
    }

    bool with_relu = param.activation_param.has_active &&
                     param.activation_param.active == Active_relu;
    float n_slope = param.activation_param.negative_slope;

    if (with_relu) {
        desc<mkldnn_relu> relu_desc = desc<mkldnn_relu>(mkldnn::prop_kind::forward_inference,
                                      mkldnn::algorithm::eltwise_relu, conv_prv_desc.dst_primitive_desc().desc(), n_slope);
        pdesc<mkldnn_relu> relu_pdesc = pdesc<mkldnn_relu>(relu_desc, *_engine);
        _prvs.push_back(mkldnn_relu(relu_pdesc, *_conv_out_mem, *_conv_out_mem));
    }
    LOG(INFO)<<"conv out mem "<<_conv_out_mem->get_primitive_desc().desc().data.format;
    LOG(INFO)<<"out mem "<<_out_mem->get_primitive_desc().desc().data.format;
    //check output_memory need reorder
    if (_conv_out_mem->get_primitive_desc() != _out_mem->get_primitive_desc()) {

        _prvs.push_back(mkldnn::reorder(*_conv_out_mem, *_out_mem));
    }

    mkldnn::stream(mkldnn::stream::kind::eager).submit(_prvs_weights_trans).wait();
    //trans weights
    //mkldnn::stream(mkldnn::stream::kind::eager).submit(weights_trans).wait();
    return SaberSuccess;
}

template <>
SaberStatus VenderDeconv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {

    return init_conv_prv(inputs, outputs, param);
}

template <>
SaberStatus VenderDeconv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus VenderDeconv2D<X86, AK_FLOAT>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    //bind data
    _in_mem->set_data_handle(inputs[0]->data());
    _out_mem->set_data_handle(outputs[0]->mutable_data());
    //submit stream
    //LOG(ERROR)<<"submitting _stream prvs";
    //_stream->submit(_prvs).wait();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(_prvs).wait();
    return SaberSuccess;
}

}
}
#endif
