#include "saber/funcs/impl/x86/saber_crop.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCrop<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CropParam<X86>& param) {
    int num = inputs[0] -> num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int offset=0;
    const float* ptr_in = (const float*)inputs[0]->data();
    float* ptr_out = (float*)outputs[0]->mutable_data();
    for(int i =0; i<num; ++i){
        offset = i*in_c*in_h*in_w;
        for(int j=_c_off;j<_c_end;++j){
            offeset += j*in_h*in_w;
            for(int k=_h_off; k<_h_end; ++k){
                offset += k*in_w;
                for(int l=_w_off; l<_w_end; ++l){
                   ptr_out[0]=ptr_in[offset+l];
                   ptr_out++;
                }
            }
        }          
    }
     

    return SaberSuccess;
}

template class SaberCrop<X86, AK_FLOAT>;
}
} // namespace anakin
