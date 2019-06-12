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

template <typename Dtype>
__global__ void ker_elt_prod(Dtype* out_data, const Dtype* in_data_a, const Dtype* in_data_b,
                                   int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data_a[tid] * in_data_b[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype>
__global__ void ker_elt_sum(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            Dtype coeff1,  Dtype coeff2, int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = coeff1 * in_data1[tid] + coeff2 * in_data2[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype>
__global__ void ker_elt_max(Dtype* out_data, const Dtype* in_data_a, const Dtype* in_data_b,
                            int count, bool with_relu) {

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

template <typename Dtype>
__global__ void ker_elt_div(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data1[tid] /in_data2[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype>
__global__ void ker_elt_with_axis_div(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            int outer_num, int mid_num, int inner_num, int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        int mid_id = (tid /inner_num) % mid_num;
        Dtype tmp = in_data1[tid] /in_data2[mid_id];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype>
__global__ void ker_elt_mul(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data1[tid] * in_data2[tid];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype>
__global__ void ker_elt_with_axis_mul(Dtype* out_data, const Dtype* in_data1, const Dtype* in_data2,
                            int outer_num, int mid_num, int inner_num, int count, bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        int mid_id = (tid /inner_num) % mid_num;
        Dtype tmp = in_data1[tid] * in_data2[mid_id];

        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype> 
__global__ void ker_elt_sum_v(Dtype* out_data, const Dtype** in_data_v, const Dtype* coeff, int in_num, int count, 
                bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = 0.f;
        for (int i = 0; i < in_num; i++) {
            tmp += coeff[i] * in_data_v[i][tid];
        }
        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }  
}

template <typename Dtype> 
__global__ void ker_elt_prod_v(Dtype* out_data, const Dtype** in_data_v,int in_num, int count, 
                bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = 1.f;
        for (int i = 0; i < in_num; i++) {
            tmp *=in_data_v[i][tid];
        }
        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype> 
__global__ void ker_elt_max_v(Dtype* out_data, const Dtype** in_data_v, int in_num, int count, 
                bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data_v[0][tid];
        for (int i = 1; i < in_num; i++) {
            tmp = in_data_v[i][tid] >  tmp ? in_data_v[i][tid] : tmp;
        }
        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <typename Dtype> 
__global__ void ker_elt_div_v(Dtype* out_data, const Dtype** in_data_v, int in_num, int count, 
                bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data_v[0][tid];
        for (int i = 1; i < in_num; i++) {
            tmp = tmp / in_data_v[i][tid];
        }
        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}


template <typename Dtype> 
__global__ void ker_elt_mul_v(Dtype* out_data, const Dtype** in_data_v, int in_num, int count, 
                bool with_relu) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype tmp = in_data_v[0][tid];
        for (int i = 1; i < in_num; i++) {
            tmp = tmp * in_data_v[i][tid];
        }
        if (with_relu) {
            out_data[tid] = tmp > static_cast<Dtype>(0.0f) ? tmp : static_cast<Dtype>(0.0f);
        } else {
            out_data[tid] = tmp;
        }
    }
}

template <>
SaberStatus SaberEltwise<NV, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<NV> *>& inputs, \
        std::vector<Tensor<NV> *>& outputs, \
        EltwiseParam<NV>& param) {
    const int count = outputs[0]->valid_size();
    float* out_data = static_cast<float*>(outputs[0]->mutable_data());
    const float* in_data_a = static_cast<float*>(inputs[0]->data());
    const float* in_data_b = static_cast<float*>(inputs[1]->data());
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int in_num = inputs.size();
    uint64_t in_data_h[in_num];
    for (int i = 0; i < in_num; i++) {
        in_data_h[i] = (uint64_t)inputs[i]->data();
    }
    uint64_t* in_data_d = (uint64_t*) _inputs_d.mutable_data();
    const float* coeff_data_d = (const float*) _coeff_d.data();
    cudaMemcpyAsync(in_data_d, in_data_h, sizeof(uint64_t) * in_num, cudaMemcpyHostToDevice, cuda_stream);


    int grid_dim = CUDA_GET_BLOCKS(count);
    int block_dim = CUDA_NUM_THREADS;
    

    switch (param.operation) {
    case Eltwise_prod:
        if (inputs.size() <= 2) {
            ker_elt_prod<float> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data, in_data_a,
                    in_data_b, count, _with_relu);
        } else {
            ker_elt_prod_v<float> <<< grid_dim, block_dim, 0, cuda_stream>>>(out_data,
                    (const float**)in_data_d,
                    in_num,
                    count, 
                    _with_relu);
        }

        break;

    case Eltwise_sum:
        if (inputs.size() <= 2) {
            ker_elt_sum <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    in_data_a, in_data_b,
                    param.coeff[0], param.coeff[1], count, _with_relu);
        } else {
            ker_elt_sum_v<float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    (const float**)in_data_d, 
                    coeff_data_d, in_num, count, _with_relu);
        }

        break;

    case Eltwise_max:
        if (inputs.size() <= 2) {
            ker_elt_max <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    in_data_a, in_data_b,
                    count, _with_relu);
        } else {
            ker_elt_max_v<float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    (const float**)in_data_d,
                     in_num,
                     count, _with_relu);
        }

        break;
    case Eltwise_div:
        if (inputs.size() <= 2) {
            if (inputs[0]->valid_size() == inputs[1]->valid_size()) {
                ker_elt_div <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                        in_data_a, in_data_b,
                        count, _with_relu);
            } else {
                int outer_num = inputs[0]->count_valid(0, param.axis);
                int mid_num = outputs[0]->valid_size();
                int inner_num = inputs[0]->count_valid(param.axis, inputs[0]->dims()) / mid_num;
                ker_elt_with_axis_div <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                        in_data_a, in_data_b, outer_num, mid_num, inner_num,
                        count, _with_relu);
            }
        } else {
            ker_elt_div_v<float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    (const float**)in_data_d, in_num, count, _with_relu);
        }

        break;

    case Eltwise_mul:
        if (inputs.size() <= 2) {
            if (inputs[0]->valid_size() == inputs[1]->valid_size()) {
                ker_elt_mul <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                        in_data_a, in_data_b,
                        count, _with_relu);
            } else {
                int outer_num = inputs[0]->count_valid(0, param.axis);
                //int mid_num = inputs[1]->valid_size();
                int mid_num = inputs[1]->count_valid(param.axis, inputs[1]->dims());
                int inner_num = inputs[0]->count_valid(param.axis, inputs[0]->dims()) / mid_num;
                ker_elt_with_axis_mul <float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                        in_data_a, in_data_b, outer_num, mid_num, inner_num,
                        count, _with_relu);
            }
        } else {
            ker_elt_mul_v<float><<<grid_dim, block_dim, 0, cuda_stream >>> (out_data,
                    (const float**)in_data_d, in_num, count, _with_relu);
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

template <>
SaberStatus SaberEltwise<NV, AK_INT8>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        EltwiseParam<NV>& param,
        Context<NV>& ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberEltwise<NV, AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        EltwiseParam<NV>& param,
        Context<NV>& ctx) {

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberEltwise<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        EltwiseParam<NV>& param) {
    return SaberSuccess;
}

template class SaberEltwise<NV, AK_FLOAT>;
template class SaberEltwise<NV, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, NV, AK_HALF);


}
}
