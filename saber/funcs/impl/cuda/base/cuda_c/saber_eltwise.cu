#include "saber/funcs/impl/cuda/saber_eltwise.h"
namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_elt_production(Dtype* out_data, const Dtype * in_data_a, const Dtype * in_data_b, int count){
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = in_data_a[tid] * in_data_b[tid];
    }
}

template <typename Dtype>
__global__ void ker_elt_sum(Dtype* out_data, const Dtype * in_data1,const Dtype * in_data2, Dtype coeff1,  Dtype coeff2, int count){
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = coeff1*in_data1[tid] + coeff2 * in_data2[tid];
    }
}

template <typename Dtype>
__global__ void ker_elt_max(Dtype * out_data, float * mask, const Dtype * in_data_a, const Dtype * in_data_b, int count, int bid){
    if(bid == 0){
        CUDA_KERNEL_LOOP(tid, count){
            Dtype var_a = in_data_a[tid];
            Dtype var_b = in_data_b[tid];
            bool a_gt_b = var_a > var_b;
            out_data[tid] = a_gt_b ? var_a : var_b;
            mask[tid] = a_gt_b ? 0 : 1;
        }
    }
    else{
        CUDA_KERNEL_LOOP(tid, count){
            Dtype var_a = in_data_a[tid];
            Dtype var_b = in_data_b[tid];
            bool a_gt_b = var_a > var_b;
            if( ! a_gt_b){
                out_data[tid] = var_b;
                mask[tid] = bid;
            }
        }
    }
}


template <>
SaberStatus SaberEltwise<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    EltwiseParam<OpTensor> &param) {
    float * mask = NULL;
    const int count = outputs[0]->size();
    float *out_data = outputs[0]->mutable_data();
    const float *in_data_a = inputs[0]->data();
	const float *in_data_b = inputs[1]->data();
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();
    switch(param.operation){
	case Eltwise_prod:
		ker_elt_production<InDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data_a,
                    in_data_b, count);

		for(int i = 2; i < inputs.size(); i++){
			ker_elt_production<InDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, out_data,
                    inputs[i]->data(), count);
		}
		break;
	case Eltwise_sum:
		ker_elt_sum<InDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                    inputs[0]->data(), inputs[1]->data(),
                    param.coeff[0], param.coeff[1], count);
		break;
	case Eltwise_max:
		mask = _max_idx.mutable_data();
		ker_elt_max<InDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, mask,
                    in_data_a, in_data_b, count, 0);

		for(int i = 2; i < inputs.size(); i++){
			ker_elt_max<InDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, mask,
                    out_data, inputs[i]->data(), count, i);
		}
		break;
	default:
		LOG(FATAL) << "unknown elementwise operation. ";
	}

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

}
}