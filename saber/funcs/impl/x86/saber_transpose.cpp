#include "saber/funcs/impl/x86/saber_transpose.h"
#include <math.h>

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberTranspose<X86, OpDtype>::dispatch(\
    const std::vector<DataTensor_in *>& inputs,\
    std::vector<DataTensor_out *>& outputs, \
    TransposeParam<X86>& param) {

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    int c_in = inputs[0]->channel();
    int n_in = inputs[0]->num();

    int num_idx = inputs[0]->num_index();
    int channel_idx = inputs[0]->channel_index();
    int height_idx = inputs[0]->height_index();
    int width_idx = inputs[0]->width_index();

    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(n_in, n_out) << "input batch size should = output batch size";
    CHECK_EQ(h_in, w_out) << "input width size should = output height size";
    CHECK_EQ(w_in, h_out) << "input height size should = output width size";

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();

    for(int k = 0; k < n_in * c_in; ++k){  
        for(int j = 0; j < h_in; ++j){
            for(int i = 0; i < w_in; ++i){
                out_data[i * w_out + j] = in_data[j * w_in + i];
            }
        }
        in_data += h_in * w_in;
        out_data += h_out * w_out;
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, X86, AK_INT8);
}//namespace saber

}//namespace anakin