#include "saber/funcs/impl/x86/saber_eltwise_act.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin{
namespace saber {

template class SaberEltwiseActive<X86, AK_FLOAT>;



template <DataType OpDtype>
void SaberEltwiseActive<X86, OpDtype>::simple_sum_relu(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseActiveParam<X86>& param) {
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target= (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);
    if(param.has_activation&&param.activation_param.active!=Active_relu){
        CHECK(false)<<"not impl";
    }
    bool is_relu=param.has_activation&&param.activation_param.active==Active_relu;
    for(int i=0;i<input_num;++i){
        in_ptrs[i]= (OpDataType*) inputs[i]->data();
    }
//TODO:can be SIMD to improve cache efficient
//TODO:can be opt by check coeff == 1
    for(int inner_id=0;inner_id<inner_size;++inner_id){
        OpDataType temp=0;
        for(int input_id=0;input_id<input_num;++input_id) {
            temp+=in_ptrs[input_id][inner_id]*param.eltwise_param.coeff[input_id];
        }
        if(is_relu&&temp<0){
            temp=0;
        }
        target[inner_id]=temp;
    }
}

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<X86, OpDtype>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseActiveParam<X86> &param, Context<X86> &ctx)
{

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<X86, OpDtype>
    ::create(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  EltwiseActiveParam<X86> &param, Context<X86> &ctx)
{
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<X86, OpDtype>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  EltwiseActiveParam<X86> &param)
{

    CHECK_EQ(outputs.size(), (size_t)1);
    if (param.eltwise_param.operation) {
        simple_sum_relu(inputs, outputs, param);
    }else{
        CHECK(false)<<"not impl";
    }
    
    return SaberSuccess;
      
}

}
} // namespace anakin
