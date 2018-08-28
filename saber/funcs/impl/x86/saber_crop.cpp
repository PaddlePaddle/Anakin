#include "saber/funcs/impl/x86/saber_crop.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCrop<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CropParam<X86>& param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    int num = inputs[0] -> num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    const DataType_in* ptr_in = (const DataType_in*)inputs[0]->data();
    DataType_out* ptr_out = (DataType_out*)outputs[0]->mutable_data();
    for(int i =0; i < num; ++i){
        int offset_n = i * in_c * in_h * in_w;
        for(int j=_c_off; j < _c_end; ++j){
            int offset_c = offset_n + j * in_h * in_w;
            for(int k=_h_off; k < _h_end; ++k){
                int offset_h = offset_c + k * in_w;
                for(int l=_w_off; l < _w_end; ++l){
                   ptr_out[0]=ptr_in[offset_h + l];
                   ptr_out++;
                }
            }
        }          
    }
    return SaberSuccess;
}

template class SaberCrop<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, X86, AK_INT8);
}
} // namespace anakin
