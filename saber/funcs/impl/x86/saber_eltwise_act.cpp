#include "saber/funcs/impl/x86/saber_eltwise_act.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin{
namespace saber {

template class SaberEltwiseActive<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <typename HTensor>
static void very_simple_eltwise_act(const std::vector<HTensor*>& inputs,
                                    std::vector<HTensor*>& outputs,
                                    EltwiseActiveParam<HTensor>& param){
    typedef typename HTensor::Dtype Dtype;
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    Dtype* target=outputs[0]->mutable_data();
    std::vector<const Dtype*> in_ptrs(input_num);
    bool is_relu=param.has_activation&&param.activation_param.active==Active_relu;
    for(int i=0;i<input_num;++i){
        in_ptrs[i]=inputs[i]->data();
    }
//TODO:can be SIMD to improve cache efficient
//TODO:can be opt by check coeff == 1
    for(int inner_id=0;inner_id<inner_size;++inner_id){
        Dtype temp=0;
        for(int input_id=0;input_id<input_num;++input_id) {
            temp+=in_ptrs[input_id][inner_id]*param.eltwise_param.coeff[input_id];
        }
        if(is_relu&&temp<0){
            target[inner_id]=0;
        }
        target[inner_id]=temp;
    }
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberEltwiseActive<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::simple_sum(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseActiveParam<OpTensor>& param) {
    const int num_arrs = inputs.size();
    const size_t nelems = inputs[0]->size();
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;
#pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        utils::balance211(blocks_number, nthr, ithr, start, end);
        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            // #pragma omp simd
            for (size_t e = start_e; e < end_e; e++) {
                outputs[0]->mutable_data()[e] = param.eltwise_param.coeff[0]
                                                * inputs[0]->mutable_data()[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                // #pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    outputs[0]->mutable_data()[e] += param.eltwise_param.coeff[a]
                                                     * inputs[a]->mutable_data()[e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            // #pragma omp simd
            for (size_t e = start_e; e < end_e; e++) {
                outputs[0]->mutable_data()[e] = param.eltwise_param.coeff[0]
                                                * inputs[0]->mutable_data()[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                // #pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    outputs[0]->mutable_data()[e] += param.eltwise_param.coeff[a]
                                                     * inputs[a]->mutable_data()[e];
                }
            }
        }
    }
}
template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberEltwiseActive<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::simple_relu(
        std::vector<DataTensor_out*>& outputs) {

    for (int i = 0; i < outputs[0]->size(); i++) {
        if (outputs[0]->mutable_data()[i] < 0) {
            outputs[0]->mutable_data()[i] = 0;
        }
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEltwiseActive<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseActiveParam<OpTensor> &param, Context<X86> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEltwiseActive<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::create(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  EltwiseActiveParam<OpTensor> &param, Context<X86> &ctx)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEltwiseActive<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  EltwiseActiveParam<OpTensor> &param)
{
#if 0
    CHECK_EQ(outputs.size(), (size_t)1);
    switch (param.eltwise_param.operation) {
        case Eltwise_sum:
            simple_sum(inputs, outputs, param);
            break;
        default:
            return SaberUnImplError;
    }

    if (param.has_activation) {
        switch (param.activation_param.active) {
            case Active_relu:
                simple_relu(outputs);
                break;
            default:
                return SaberUnImplError;
        }
    }
#endif
    CHECK_EQ(outputs.size(), (size_t)1);
    if (param.eltwise_param.operation) {
        very_simple_eltwise_act(inputs, outputs, param);
    }else{
        CHECK(false)<<"not impl";
    }
    return SaberSuccess;
      
}

}
} // namespace anakin
