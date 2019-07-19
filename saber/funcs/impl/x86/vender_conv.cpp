#include "anakin_config.h"
#ifndef USE_SGX
#include "saber/funcs/impl/x86/vender_conv.h"

namespace anakin {
namespace saber {

template <DataType Dtype>
SaberStatus VenderConv2D<X86, Dtype>::init_conv_prv_any(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs, ConvParam<X86>& param){

    _engine = std::make_shared<mkldnn::engine>(mkldnn::engine::cpu, 0);
    _alg = mkldnn::algorithm::convolution_direct;
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
    bool with_dilation = (param.dilation_w == 1 && param.dilation_h == 1)? false : true;

    //TODO:here we ignored group
    std::shared_ptr<desc<mkldnn_conv> > conv_desc;
    if (with_bias && with_dilation){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, bias_md, out_md, strides, dilation, padding, padding,
            mkldnn::padding_kind::zero);
    } else if (with_bias){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, bias_md, out_md, strides, padding, padding,
            mkldnn::padding_kind::zero);
    } else if (with_dilation){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, out_md, strides, dilation, padding, padding,
            mkldnn::padding_kind::zero);
    } else {
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, out_md, strides, padding, padding,
            mkldnn::padding_kind::zero);
    }

    pdesc<mkldnn_conv> conv_prv_desc = pdesc<mkldnn_conv>(*conv_desc, *_engine);

    //above: make convolution_primitive_description
    //below: make memorys

    //make input_memory and weights_memory for user
    _in_mem = create_mkldnn_memory_no_data(inputs[0], *_engine);
    _w_mem = create_mkldnn_memory(param.mutable_weight(), w_sh, 
        mkldnn_mem_format::oihw, mkldnn_mem_dtype::f32, *_engine);

    //set input_memory and weights_memory for conv
    _conv_in_mem = _in_mem;
    if (pdesc<mkldnn_mem>(conv_prv_desc.src_primitive_desc()) != _in_mem->get_primitive_desc()){
        _conv_in_mem.reset(new mkldnn_mem(conv_prv_desc.src_primitive_desc()));
        _prvs.push_back(mkldnn::reorder(*_in_mem, *_conv_in_mem));
    }
    //std::vector<mkldnn::primitive> weights_trans;
    _conv_w_mem = _w_mem;
    if (pdesc<mkldnn_mem>(conv_prv_desc.weights_primitive_desc()) != _w_mem->get_primitive_desc()){
        _conv_w_mem.reset(new mkldnn_mem(conv_prv_desc.weights_primitive_desc()));
        
        //weights_trans.push_back(mkldnn::reorder(w_mem, conv_w_mem));
        _prvs.push_back(mkldnn::reorder(*_w_mem, *_conv_w_mem));
    }
    
    //set output_memory for user and conv
    _out_mem = create_mkldnn_memory_no_data(outputs[0], *_engine);
    _conv_out_mem = _out_mem;
    if (pdesc<mkldnn_mem>(conv_prv_desc.dst_primitive_desc()) != _out_mem->get_primitive_desc()){
        _conv_out_mem.reset(new mkldnn_mem(conv_prv_desc.dst_primitive_desc()));
    }

    //set bias_memory for user and conv
    //make convolution primitive 
    if (with_bias){
        _bias_mem = create_mkldnn_memory(param.mutable_bias(), b_sh, 
            mkldnn_mem_format::x, mkldnn_mem_dtype::f32, *_engine);
        _conv_bias_mem = _bias_mem;
        if (pdesc<mkldnn_mem>(conv_prv_desc.bias_primitive_desc()) != _bias_mem->get_primitive_desc()){
            _conv_bias_mem.reset(new mkldnn_mem(conv_prv_desc.bias_primitive_desc()));
            _prvs.push_back(mkldnn::reorder(*_bias_mem, *_conv_bias_mem));    
        }

        _prvs.push_back(mkldnn_conv(conv_prv_desc, *_conv_in_mem, *_conv_w_mem, *_conv_bias_mem, *_conv_out_mem));
    } else {
        _prvs.push_back(mkldnn_conv(conv_prv_desc, *_conv_in_mem, *_conv_w_mem, *_conv_out_mem));
    }

    bool with_relu = param.activation_param.has_active &&
     param.activation_param.active == Active_relu;
    float n_slope = param.activation_param.negative_slope;
    if (with_relu){
        desc<mkldnn_relu> relu_desc = desc<mkldnn_relu>(mkldnn::prop_kind::forward_inference,
            mkldnn::algorithm::eltwise_relu, conv_prv_desc.dst_primitive_desc().desc(), n_slope);
        pdesc<mkldnn_relu> relu_pdesc = pdesc<mkldnn_relu>(relu_desc, *_engine);
        _prvs.push_back(mkldnn_relu(relu_pdesc, *_conv_out_mem, *_conv_out_mem)); 
    }

    //check output_memory need reorder
    if (_conv_out_mem->get_primitive_desc() != _out_mem->get_primitive_desc()){
        _prvs.push_back(mkldnn::reorder(*_conv_out_mem, *_out_mem));
    }

    //trans weights
    //mkldnn::stream(mkldnn::stream::kind::eager).submit(weights_trans).wait();
    return SaberSuccess;

}

template <DataType Dtype>
SaberStatus VenderConv2D<X86, Dtype>::init_conv_prv_specify(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs, ConvParam<X86>& param){

    _engine = std::make_shared<mkldnn::engine>(mkldnn::engine::cpu, 0);
    _alg = mkldnn::algorithm::convolution_direct;

    Shape in_sh = inputs[0]->valid_shape();
    Shape out_sh = outputs[0]->valid_shape();
    std::vector<int> b_sh = {out_sh.channel()};
    std::vector<int> w_sh = param.weight()->valid_shape(); 

    auto in_md = create_mkldnn_memory_desc(in_sh, 
        get_mkldnn_dtype(inputs[0]->get_dtype()), get_mkldnn_format(inputs[0]->get_layout()));
    auto bias_md = create_mkldnn_memory_desc(b_sh, 
        get_mkldnn_dtype(inputs[0]->get_dtype()), mkldnn_mem_format::x);
    auto weights_md = create_mkldnn_memory_desc<Dtype>(w_sh);
    auto out_md = create_mkldnn_memory_desc(out_sh, 
        get_mkldnn_dtype(outputs[0]->get_dtype()), get_mkldnn_format(outputs[0]->get_layout()));

    mkldnn_mem_dim strides = {param.stride_h, param.stride_w};
    mkldnn_mem_dim dilation = {param.dilation_h, param.dilation_w};
    mkldnn_mem_dim padding = {param.pad_h, param.pad_w};

    bool with_bias = param.bias() && param.bias() -> valid_size() > 0 ? true : false;
    bool with_dilation = (param.dilation_w == 1 && param.dilation_h == 1)? false : true;

    //TODO:here we ignored group
    std::shared_ptr<desc<mkldnn_conv> > conv_desc;
    if (with_bias && with_dilation){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, bias_md, out_md, strides, dilation, padding, padding,
            mkldnn::padding_kind::zero);
    } else if (with_bias){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, bias_md, out_md, strides, padding, padding,
            mkldnn::padding_kind::zero);
    } else if (with_dilation){
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, out_md, strides, dilation, padding, padding,
            mkldnn::padding_kind::zero);
    } else {
        conv_desc = std::make_shared<desc<mkldnn_conv> >(mkldnn::prop_kind::forward_inference, _alg,
            in_md, weights_md, out_md, strides, padding, padding,
            mkldnn::padding_kind::zero);
    }

    pdesc<mkldnn_conv> conv_prv_desc = pdesc<mkldnn_conv>(*conv_desc, *_engine);
    //above: make convolution_primitive_description
    //below: make memorys

    //make input_memory and weights_memory for user
    _in_mem = create_mkldnn_memory_no_data(inputs[0], *_engine);
    _w_mem = create_mkldnn_memory(param.mutable_weight(), w_sh, 
        get_mkldnn_format(param.weight()->get_layout()),
        get_mkldnn_dtype(param.weight()->get_dtype()), *_engine);
    
    //set output_memory for user and conv
    _out_mem = create_mkldnn_memory_no_data(outputs[0], *_engine);

    //set bias_memory for user and conv
    //make convolution primitive 
    _conv_w_mem = _w_mem;
    if (pdesc<mkldnn_mem>(conv_prv_desc.weights_primitive_desc()) != _w_mem->get_primitive_desc()){
        _conv_w_mem.reset(new mkldnn_mem(conv_prv_desc.weights_primitive_desc()));
        
        //weights_trans.push_back(mkldnn::reorder(w_mem, conv_w_mem));
        _pre_prvs.push_back(mkldnn::reorder(*_w_mem, *_conv_w_mem));
    }

    if (with_bias){
        _bias_mem = create_mkldnn_memory(param.mutable_bias(), b_sh, 
            mkldnn_mem_format::x, get_mkldnn_dtype(param.bias()->get_dtype()), *_engine);

        _prvs.push_back(mkldnn_conv(conv_prv_desc, *_in_mem, *_conv_w_mem, *_bias_mem, *_out_mem));
    } else {
        _prvs.push_back(mkldnn_conv(conv_prv_desc, *_in_mem, *_conv_w_mem, *_out_mem));
    }

    bool with_relu = param.activation_param.has_active &&
     param.activation_param.active == Active_relu;
    float n_slope = param.activation_param.negative_slope;
    if (with_relu){
        desc<mkldnn_relu> relu_desc = desc<mkldnn_relu>(mkldnn::prop_kind::forward_inference,
            mkldnn::algorithm::eltwise_relu, conv_prv_desc.dst_primitive_desc().desc(), n_slope);
        pdesc<mkldnn_relu> relu_pdesc = pdesc<mkldnn_relu>(relu_desc, *_engine);
        _prvs.push_back(mkldnn_relu(relu_pdesc, *_out_mem, *_out_mem)); 
    }

    //trans weights
    mkldnn::stream(mkldnn::stream::kind::eager).submit(_pre_prvs).wait();
    return SaberSuccess;

}

template <>
SaberStatus VenderConv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {

    //init_conv_prv_any(inputs, outputs, param);
    return init_conv_prv_specify(inputs, outputs, param);
}

template <>
SaberStatus VenderConv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    if(param.group>1){
        return SaberUnImplError;
    }
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus VenderConv2D<X86, AK_FLOAT>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    if(param.group>1){
        return SaberUnImplError;
    }
    //bind data
    _in_mem->set_data_handle(inputs[0]->data());
    _out_mem->set_data_handle(outputs[0]->mutable_data());
    //submit stream
    mkldnn::stream(mkldnn::stream::kind::eager).submit(_prvs).wait();
    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(VenderConv2D, ConvParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(VenderConv2D, ConvParam, X86, AK_INT8);

}
}
#endif
