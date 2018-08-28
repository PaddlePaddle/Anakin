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
/*
template <DataType OpDtype>
void SaberEltwise<X86, OpDtype>::simple_sum(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param){

    const int num_arrs = inputs.size();
    const size_t nelems = inputs[0]->size();
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;
    OpDataType* out=(OpDataType*) outputs[0]->mutable_data();
    	
#pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        utils::balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
             //#pragma omp simd
            for (size_t e = start_e; e < end_e; e++) {
                 out[e] = param.coeff[0] * ((OpDataType*) inputs[0]->mutable_data())[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                 //#pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                   out[e] += param.coeff[a] * ((OpDataType*) inputs[a]->mutable_data())[e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
             //#pragma omp simd
            for (size_t e = start_e; e < end_e; e++) {
                 out[e] = param.coeff[0] * ((OpDataType*) inputs[0]->mutable_data())[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                // #pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                     out[e] += param.coeff[a] * ((OpDataType*) inputs[a]->mutable_data())[e];
                }
            }
        }
    }
}
*/
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
            return SaberSuccess;
        case Eltwise_prod:
            simple_prod(inputs, outputs, param);
            return SaberSuccess;
        case Eltwise_max:
            simple_max(inputs, outputs, param);
            return SaberSuccess;
        default:
            return SaberUnImplError;
    }
      
}

}
} // namespace anakin
