#include "saber/funcs/impl/x86/saber_pad.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberPad<X86, OpDtype>::dispatch(\
        const std::vector<Tensor<X86> *>& inputs, \
        std::vector<Tensor<X86> *>& outputs, \
        PadParam<X86>& param) {

    const dtype* in_data = static_cast<const dtype*>(inputs[0]->data());
    dtype* out_data = static_cast<dtype*>(outputs[0]->mutable_data());
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();
    int out_n = out_shape.num();
    int out_c = out_shape.channel();
    int out_h = out_shape.height();
    int out_w = out_shape.width();
    int pad_h_top = param.pad_h[0];
    int pad_h_bottom = param.pad_h[1];
    int pad_w_left = param.pad_w[0];
    int pad_w_right = param.pad_w[1];
    int pad_c_0 = param.pad_c[0];
    int pad_c_1 = param.pad_c[1];

    int ceil_in_c = in_shape.channel();
    int ceil_in_h = in_shape.height();
    int ceil_in_w = in_shape.width();



    for (size_t n_index = 0; n_index < out_n; n_index++) {
        for (size_t c_index = 0; c_index < out_c; c_index++) {
            int c_in_index = c_index - pad_c_0;
            bool is_pad_c = c_in_index < 0 || c_in_index >= ceil_in_c;
            for (size_t h_index = 0; h_index < out_h; h_index++) {
                int h_in_index = h_index - pad_h_top;
                bool is_pad_h =  h_in_index < 0 || h_in_index >= ceil_in_h;
                for (size_t w_index = 0; w_index < out_w; w_index++) {
                    int w_in_index = w_index - pad_w_left;
                    bool is_pad_w =  w_in_index < 0 || w_in_index >= ceil_in_w;
                    bool is_pad = is_pad_c||is_pad_h||is_pad_w;
                    int in_index = n_index * _in_n_stride + c_in_index * _in_c_stride + h_in_index * _in_h_stride +
                                   w_in_index * _in_w_stride;
                    int out_index = n_index * _out_n_stride + c_index * _out_c_stride + h_index * _out_h_stride +
                                    w_index * _out_w_stride;
//                    LOG(INFO)<<in_index<<","<<out_index<<","<<is_pad<<"::"<<c_in_index<<","<<h_in_index<<","<<w_in_index<<"::"<<ceil_in_c<<","<<ceil_in_h<<","<<ceil_in_w<<":::"
//                             <<(c_in_index < 0 || c_in_index >= ceil_in_c)<<","<<(h_in_index < 0 || h_in_index >= ceil_in_h)<<","<<(w_in_index < 0 || w_in_index >= ceil_in_w);
                    if (is_pad) {
                        out_data[out_index] = 0;
                    } else {
                        out_data[out_index] = in_data[in_index];
                    }
                }
            }
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPad, PadParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPad, PadParam, X86, AK_INT8);

}
}