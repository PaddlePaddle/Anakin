#include "saber/funcs/impl/x86/saber_crop.h"
//#include "mkl_cblas.h"
//#include "mkl_vml_functions.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCrop<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CropParam<X86>& param) {
        	
//            LOG(INFO)<<"here!!!";
    int num = inputs[0] -> num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int count = outputs[0]->valid_size();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    float* ptr_in = (float*)inputs[0]->data();
    float* ptr_out = (float*)outputs[0]->mutable_data();
    for(int i =0;i<num;++i){
        for(int j=_c_off;j<_c_end;++j){
            for(int k=_h_off;k<_h_end;++k){
                for(int l=_w_off;l<_w_end;++l){
                   //LOG(INFO)<<k*in_w+l;
                   LOG(INFO)<<"c_off="<<_c_end;
                   LOG(INFO)<<"_h_off="<<_h_end;
                                                         LOG(INFO)<<"w_off="<<_w_end;
                   ptr_out[0]=ptr_in[i*in_c*in_h*in_w+j*in_h*in_w+k*in_w+l];
                   LOG(INFO)<<"in="<<ptr_in[i*in_c*in_h*in_w+j*in_h*in_w+k*in_w+l];
                   LOG(INFO)<<"off="<<i*in_c*in_h*in_w+j*in_h*in_w+k*in_w+l;
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
