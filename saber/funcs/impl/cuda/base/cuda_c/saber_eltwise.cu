#include "saber/funcs/impl/cuda/saber_eltwise.h"
namespace anakin {
namespace saber {
#if 0
template <typename Dtype, bool with_relu>
static __global__ void ker_multi_elt_production(Dtype* out_data, const Dtype** in_data, int count,
        int input_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        Dtype tmp = in_data[0][tid];

        for (int i = 1; i < input_size; i++) {
            tmp *= in_data[i][tid];
        }

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0) ? tmp : static_cast<Dtype>(0);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype, bool with_relu>
static __global__ void ker_multi_elt_sum(Dtype* out_data, const Dtype** in_data, const Dtype* coeff,
        int count, int input_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        Dtype tmp = coeff[0] * in_data[0][tid];

        for (int i = 1; i < input_size; i++) {
            tmp += coeff[i] * in_data[i][tid];
        }

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype, bool with_relu>
static __global__ void ker_multi_elt_max(Dtype* out_data, const Dtype** in_data, int count,
        int input_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        Dtype tmp = in_data[0][tid];

        for (int i = 1; i < input_size; i++) {
            tmp = tmp >= in_data[i][tid] ? tmp : in_data[i][tid];
        }

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0) ? tmp : static_cast<Dtype>(0);
        } else {
            out_data[tid] = tmp;
        }
    }
}
#endif

template <typename Dtype, bool with_relu>
__global__ void ker_elt_production(Dtype* out_data, const Dtype* in_data_a, const Dtype* in_data_b,
                                   int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data_a[tid] * in_data_b[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype, bool with_relu>
__global__ void ker_elt_sum(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            Dtype coeff1,  Dtype coeff2, int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = coeff1 * in_data1[tid] + coeff2 * in_data2[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype, bool with_relu>
__global__ void ker_elt_max(Dtype* out_data, const Dtype* in_data_a, const Dtype* in_data_b,
                            int count) {

    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp;
        Dtype var_a = in_data_a[tid];
        Dtype var_b = in_data_b[tid];
        bool a_gt_b = var_a > var_b;
        tmp = a_gt_b ? var_a : var_b;

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}


template <DataType OpDtype>
SaberStatus SaberEltwise<NV, OpDtype>::dispatch(\
        const std::vector<Tensor<NV> *>& inputs, \
        std::vector<Tensor<NV> *>& outputs, \
        EltwiseParam<NV>& param) {
    const int count = outputs[0]->valid_size();
    OpDataType* out_data = static_cast<OpDataType*>(outputs[0]->mutable_data());
    const OpDataType* in_data_a = static_cast<OpDataType*>(inputs[0]->data());
    const OpDataType* in_data_b = static_cast<OpDataType*>(inputs[1]->data());
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();


    int grid_dim = CUDA_GET_BLOCKS(count);
    int block_dim = CUDA_NUM_THREADS;

    switch (param.operation) {
    case Eltwise_prod:
        if (_with_relu) {
            if (inputs.size() <= 2) {
                ker_elt_production<OpDataType, true> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, in_data_a,
                        in_data_b, count);
            } else {
                ker_elt_production<OpDataType, false> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data,
                        in_data_a,
                        in_data_b, count);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    ker_elt_production<OpDataType, false>
                    <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                            static_cast<const OpDataType*>(inputs[i]->data()), count);
                }

                ker_elt_production<OpDataType, true>
                <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                        static_cast<const OpDataType*>(inputs[inputs.size() - 1]->data()), count);
            }

        } else {

            ker_elt_production<OpDataType, false> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data,
                    in_data_a,
                    in_data_b, count);

            for (int i = 2; i < inputs.size(); i++) {
                ker_elt_production<OpDataType, false>
                <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                        static_cast<const OpDataType*>(inputs[i]->data()), count);
            }

        }

        break;

    case Eltwise_sum:
        if (_with_relu) {
            ker_elt_sum <OpDataType, true>
            <<<
            grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    in_data_a, in_data_b,
                    param.coeff[0], param.coeff[1], count);
        } else {
            ker_elt_sum <OpDataType, false>
            <<<
            grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    in_data_a, in_data_b,
                    param.coeff[0], param.coeff[1], count);
        }

        break;

    case Eltwise_max:

        //      mask = (float *) _max_idx.mutable_data();
        if (_with_relu) {
            if (inputs.size() <= 2) {
                ker_elt_max<OpDataType, true>
                <<< grid_dim, block_dim, 0, cuda_stream >>>(out_data,
                        in_data_a, in_data_b, count);
            } else {
                ker_elt_max<OpDataType, false> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data,
                        in_data_a,
                        in_data_b, count);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    ker_elt_max<OpDataType, false>
                    <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                            static_cast<const OpDataType*>(inputs[i]->data()), count);
                }

                ker_elt_max<OpDataType, true>
                <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                        static_cast<const OpDataType*>(inputs[inputs.size() - 1]->data()), count);
            }
        } else {

            ker_elt_max<OpDataType, false> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data,
                    in_data_a,
                    in_data_b, count);

            for (int i = 2; i < inputs.size() ; i++) {
                ker_elt_max<OpDataType, false>
                <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, out_data,
                        static_cast<const OpDataType*>(inputs[i]->data()), count);
            }

        }


        break;

    default:
        LOG(FATAL) << "unknown elementwise operation. ";
    }

    if (_other_activation) {
        SABER_CHECK(_saber_activation.dispatch(inputs, outputs, param.activation_param));
    }

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberEltwise<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, NV, AK_INT8);

}
}