#include "saber/funcs/impl/x86/saber_eltwise.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin{
namespace saber {

template class SaberEltwise<X86, AK_FLOAT>;

template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86> &param,
        Context<X86> &ctx)
{
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param,
        Context<X86> &ctx)
{
    this->_param = &param;
    this->_ctx = &ctx;

    return SaberSuccess;
}
template <DataType OpDtype>
void SaberEltwise<X86, OpDtype>::simple_sum(const std::vector<DataTensor_in*>& inputs,
                                    std::vector<DataTensor_out*>& outputs,
                                    EltwiseParam<X86>& param){
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target= (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);
    for(int i = 0; i < input_num; ++i){
        in_ptrs[i] = ( OpDataType* ) inputs[i]->data();
    }
//TODO:can be SIMD to improve cache efficient
    for(int inner_id = 0; inner_id < inner_size; ++inner_id){
        target[inner_id] = in_ptrs[0][inner_id]* param.coeff[0];
    }
    for(int input_id = 1; input_id < input_num; ++input_id){
        for(int inner_id = 0; inner_id < inner_size; ++inner_id){
            target[inner_id] += in_ptrs[input_id][inner_id]* param.coeff[input_id];
        }
    }
}
template <DataType OpDtype>
void SaberEltwise<X86, OpDtype>::simple_prod(const std::vector<DataTensor_in*>& inputs,
                                    std::vector<DataTensor_out*>& outputs,
                                    EltwiseParam<X86>& param){
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target= (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);
    for(int i = 0; i < input_num; ++i){
        in_ptrs[i] = ( OpDataType* ) inputs[i]->data();
    }
//TODO:can be SIMD to improve cache efficient
    for(int inner_id = 0; inner_id < inner_size; ++inner_id){
        target[inner_id] = in_ptrs[0][inner_id];
    }
    for(int input_id = 1; input_id < input_num; ++input_id){
        for(int inner_id = 0; inner_id < inner_size; ++inner_id){
            target[inner_id] *= in_ptrs[input_id][inner_id];
        }
    }
}

template <DataType OpDtype>
void SaberEltwise<X86, OpDtype>::simple_max(const std::vector<DataTensor_in*>& inputs,
                                    std::vector<DataTensor_out*>& outputs,
                                    EltwiseParam<X86>& param){
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target= (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);
    for(int i = 0; i < input_num; ++i){
        in_ptrs[i] = ( OpDataType* ) inputs[i]->data();
    }
//TODO:can be SIMD to improve cache efficient
    for(int inner_id = 0; inner_id < inner_size; ++inner_id){
        target[inner_id] = in_ptrs[0][inner_id];
    }
    for(int input_id = 1; input_id < input_num; ++input_id){
        for(int inner_id = 0; inner_id < inner_size; ++inner_id){
            target[inner_id] = target[inner_id] > in_ptrs[input_id][inner_id] ? target[inner_id] : in_ptrs[input_id][inner_id];
        }
    }
}
    
template <DataType OpDtype>
void SaberEltwise<X86, OpDtype>::simple_relu(std::vector<DataTensor_in*>& inputs){
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    std::vector<OpDataType*> in_ptrs(input_num);
    for(int i = 0; i < input_num; ++i){
        in_ptrs[i] = ( OpDataType* ) inputs[i]->mutable_data();
    }
//TODO:can be SIMD to improve cache efficient
    for(int input_id = 0; input_id < input_num; ++input_id){
        for(int inner_id = 0; inner_id < inner_size; ++inner_id){
            in_ptrs[input_id][inner_id] = in_ptrs[input_id][inner_id] > 0.0f ? in_ptrs[input_id][inner_id]  : 0.0f;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86> &param)
{
    CHECK_EQ(outputs.size(), (size_t)1);
    switch (param.operation) {
        case Eltwise_sum:
            simple_sum(inputs, outputs, param);
            break;
        case Eltwise_prod:
            simple_prod(inputs, outputs, param);
            break;
        case Eltwise_max:
            simple_max(inputs, outputs, param);
            break;
        default:
		  LOG(FATAL) << "unknown elementwise operation. ";
    }
    if(param.activation_param.has_active){
        switch (param.activation_param.active) {
            case Active_relu:
                simple_relu(outputs);
                break;
            default:
                LOG(FATAL) << "unknown elementwise active operation. ";;
        }
    }
    return SaberSuccess;
    
      
}

}
} // namespace anakin
