#include "saber/funcs/impl/cuda/saber_eltwise_act.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_elt_sum_with_relu(Dtype* out_data, const Dtype * in_data1,const Dtype * in_data2,  Dtype coeff1,Dtype coeff2, int count){
    CUDA_KERNEL_LOOP(tid, count){
        Dtype temp = coeff1*in_data1[tid] + coeff2 * in_data2[tid];
        out_data[tid] = temp > 0.0 ? temp : 0.0;
    }
}

template <>
SaberStatus SaberEltwiseActive<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    EltwiseActiveParam<OpTensor> &param) {

    const int count = outputs[0]->size();
    float * out_data = outputs[0]->mutable_data();
    const float *in_data_a = inputs[0]->data();
    const float *in_data_b = inputs[1]->data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    switch(param.eltwise_param.operation){
        case Eltwise_prod:
            LOG(FATAL)<<"NOT IMPLEMENT yet!!";

            break;
        case Eltwise_sum:
            ker_elt_sum_with_relu<InDataType>
                    <<<CUDA_GET_BLOCKS(count),
                    CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                            inputs[0]->data(), inputs[1]->data(),
                            param.eltwise_param.coeff[0], param.eltwise_param.coeff[1], count);
            break;
        case Eltwise_max:
            LOG(FATAL)<<"NOT IMPLEMENT yet!!";

            break;
        default:
        LOG(FATAL) << "unknown elementwise operation. ";
    }

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

}
}