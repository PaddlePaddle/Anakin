#include "saber/funcs/impl/x86/saber_permute_power.h"

namespace anakin{
namespace saber{

template class SaberPermutePower<X86, AK_FLOAT>;
    
template <>
SaberStatus SaberPermutePower::\
    dispatch(const std::vector<Tensor<X86>*>& inputs,
                    std::vector<Tensor<X86>*>& outputs,
             PermutePowerParam<X86> &permute_param){
        
}

    
}
}
