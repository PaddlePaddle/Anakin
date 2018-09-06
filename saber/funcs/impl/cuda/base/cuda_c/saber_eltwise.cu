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
__global__ void ker_elt_sum_add(Dtype* out_data, const Dtype * in_data,  Dtype coeff, int count){
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] += coeff*in_data[tid];
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

template <typename Dtype>
__global__ void ker_relu(Dtype* out_data, int count){
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = out_data[tid] > 0.0f ? out_data[tid] : 0.0f;
    }
}

template <DataType OpDtype>
SaberStatus SaberEltwise<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    EltwiseParam<NV> &param) {
    float * mask = NULL;
    const int count = outputs[0]->size();
    OpDataType *out_data = (OpDataType *) outputs[0]->mutable_data();
    const OpDataType *in_data_a = (OpDataType *) inputs[0]->data();
    const OpDataType *in_data_b = (OpDataType *) inputs[1]->data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    switch(param.operation){
	case Eltwise_prod:
		ker_elt_production<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data_a,
                    in_data_b, count);

		for(int i = 2; i < inputs.size(); i++){
			ker_elt_production<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, out_data,
                    (OpDataType *)inputs[i]->data(), count);
		}
		break;
	case Eltwise_sum:
		ker_elt_sum<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                    in_data_a, in_data_b,
                    param.coeff[0], param.coeff[1], count);
          for(int i = 2; i < inputs.size(); i++){
			ker_elt_sum_add<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                    (OpDataType *)inputs[i]->data() ,param.coeff[i], count);
		}
		break;
	case Eltwise_max:
		mask = (float *) _max_idx.mutable_data();
		ker_elt_max<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, mask,
                    in_data_a, in_data_b, count, 0);

		for(int i = 2; i < inputs.size(); i++){
			ker_elt_max<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, mask,
                    out_data, (OpDataType *)inputs[i]->data(), count, i);
		}
		break;
	default:
		LOG(FATAL) << "unknown elementwise operation. ";
	}
    if(param.activation_param.has_active){
        switch (param.activation_param.active) {
            case Active_relu:
                ker_relu<OpDataType>
            <<<CUDA_GET_BLOCKS(count),
            CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, count);
                break;
            default:
                LOG(FATAL) << "unknown elementwise active operation. ";
        }
    }
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

}
}