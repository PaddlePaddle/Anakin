#include "saber/funcs/impl/x86/saber_cast.h"

namespace anakin{

namespace saber{

template <typename Dtype, typename Ttype>
void cast_kernel(const Dtype* src, Ttype* dst, int count) {
    for (int i = 0; i < count; i++){
        dst[i] = static_cast<Ttype>(src[i]);
    }
}

template <DataType OpDtype>
SaberStatus SaberCast<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs, CastParam<X86> &param) {

    int count = inputs[0]->valid_size();
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    if(_inDtype == _outDtype){
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    if(inputs[0]->get_dtype() == 1){//AK_FLOAT
        const float* in_data = (const float*)inputs[0]->data();
        int* out_data = (int*)outputs[0]->mutable_data();
        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
            cast_kernel<float, int>(in_data, out_data, count);
        }
        
    }
    
    if(inputs[0]->get_dtype() == 5){//AK_INT32
        const int* in_data = (const int*)inputs[0]->data();
        float* out_data = (float*)outputs[0]->mutable_data();
        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
            cast_kernel<int, float>(in_data, out_data, count);
        }
    }
   
    return SaberSuccess;
}

template class SaberCast<X86, AK_FLOAT>;
template class SaberCast<X86, AK_INT32>;
DEFINE_OP_TEMPLATE(SaberCast, CastParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCast, CastParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
