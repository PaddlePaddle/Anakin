
#include "saber/funcs/impl/cuda/saber_reduce.h"
#include "saber/funcs/impl/cuda/vender_reduce.h"
namespace anakin {
namespace saber {
namespace {
template <ReduceType type>
class ReOp {
public:
    __device__
    static float compute(float a, float b) {
        return -1.f;
    }
};
template <>
__device__
float ReOp<Reduce_max>::compute(float a, float b) {
    return ((a > b) ? a : b);
}

template <>
__device__
float ReOp<Reduce_min>::compute(float a, float b) {
    return ((a > b) ? b : a);
}

template <>
__device__
float ReOp<Reduce_sum>::compute(float a, float b) {
    return a + b;
}

template <>
__device__
float ReOp<Reduce_avg>::compute(float a, float b) {
    return a + b;
}

template <>
__device__
float ReOp<Reduce_prod>::compute(float a, float b) {
    return a * b;
}

template <int nDim>
class IndexCompute {
public:
    __device__
    static int input_idx(const int* dims,
                         const int* odims,
                         int out_idx);
};

template <>
__device__
int IndexCompute<4>::input_idx(
        const int* in_stride,
        const int* out_stride,
        int out_idx) {

    int i0 = out_idx / out_stride[0];
    int i1 = (out_idx % out_stride[0]) / out_stride[1];
    int i2 = (out_idx % out_stride[1]) / out_stride[2];
    int i3 = (out_idx % out_stride[2]) / out_stride[3];
    int idx = i0 * in_stride[0]
              + i1 * in_stride[1]
              + i2 * in_stride[2]
              + i3 * in_stride[3];
    return idx;
}

template <>
__device__
int IndexCompute<3>::input_idx(
        const int* in_stride,
        const int* out_stride,
        int out_idx) {

    int i0 = out_idx / out_stride[0];
    int i1 = (out_idx % out_stride[0]) / out_stride[1];
    int i2 = (out_idx % out_stride[1]) / out_stride[2];
    int idx = i0 * in_stride[0]
              + i1 * in_stride[1]
              + i2 * in_stride[2];
    return idx;
}

template <>
__device__
int IndexCompute<2>::input_idx(
        const int* in_stride,
        const int* out_stride,
        int out_idx) {

    int i0 = out_idx / out_stride[0];
    int i1 = (out_idx % out_stride[0]) / out_stride[1];
    int idx = i0 * in_stride[0]
              + i1 * in_stride[1];
    return idx;
}

template <>
__device__
int IndexCompute<1>::input_idx(
        const int* in_stride,
        const int* out_stride,
        int out_idx) {

    int i0 = out_idx / out_stride[0];
    int idx = i0 * in_stride[0];
    return idx;
}

// if you are reading this, there are still a lot
// optimize here to do, This class is the right class
// to make parallel reduction.
// the compute function can run inside one block,
// try to use shuffle instruction here.
// int tdim is the threads num of one block.
template <int rdim, int tdim, ReduceType type>
class ReduceCompute{
public:
    __device__
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float* in_data, int in_idx) {
        return 0;
    }
};

template <int tdim, ReduceType type>
class ReduceCompute<1, tdim, type> {
public:
    __device__
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float *in_data, int in_idx) {

//        int tid = threadIdx.x;
        float res = in_data[in_idx];
        int idx = in_idx + in_stride[rdims[0]];
        // here is the reduction op.
        for (int i = 1; i < dims[rdims[0]]; ++i) {
            res = ReOp<type>::compute(res, in_data[idx]);
            idx += in_stride[rdims[0]];
        }
        return res;
    }
};

template <int tdim, ReduceType type>
class ReduceCompute<2, tdim, type> {
public:
    __device__
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float *in_data, int in_idx) {

        float res0 = 0.f;
        int idx0 = in_idx;
        for (int i = 0; i < dims[rdims[0]]; ++i) {
            float res1 = in_data[idx0];
            int idx1 = idx0 + in_stride[rdims[1]];
            for (int j = 1; j < dims[rdims[1]]; ++j) {
                res1 = ReOp<type>::compute(res1, in_data[idx1]);
                idx1 += in_stride[rdims[1]];
            }
            idx0 += in_stride[rdims[0]];
            if (i == 0) {
                res0 = res1;
            } else {
                res0 = ReOp<type>::compute(res0, res1);
            }
        }
        return res0;
    }
};

template <int tdim, ReduceType type>
class ReduceCompute<3, tdim, type> {
public:
    __device__
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float *in_data, int in_idx) {

        float res0 = 0.f;
        int idx0 = in_idx;
        for (int i = 0; i < dims[rdims[0]]; ++i) {
            float res1 = 0.f;
            int idx1 = idx0;
            for (int j = 0; j < dims[rdims[1]]; ++j) {
                float res2 = in_data[idx1];
                int idx2 = idx1 + in_stride[rdims[2]];
                for (int k = 1; k < dims[rdims[2]]; ++k) {
                    res2 = ReOp<type>::compute(res2, in_data[idx2]);
                    idx2 += in_stride[rdims[2]];
                }
                if (j == 0) {
                    res1 = res2;
                } else {
                    res1 = ReOp<type>::compute(res1, res2);
                }
                idx1 += in_stride[rdims[1]];
            }
            if (i == 0) {
                res0 = res1;
            } else {
                res0 = ReOp<type>::compute(res0, res1);
            }
            idx0 += in_stride[rdims[0]];
        }
        return res0;
    }
};

template <int tdim, ReduceType type>
class ReduceCompute<4, tdim, type> {
public:
    __device__
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float *in_data, int in_idx) {

        float res0 = 0.f;
        int idx0 = in_idx;
        for (int i = 0; i < dims[rdims[0]]; ++i) {
            float res1 = 0.f;
            int idx1 = idx0;
            for (int j = 0; j < dims[rdims[1]]; ++j) {
                float res2 = 0.f;
                int idx2 = idx1;
                for (int k = 0; k < dims[rdims[2]]; ++k) {
                    float res3 = in_data[idx2];
                    int idx3 = idx2 + in_stride[rdims[3]];
                    for (int u = 0; u < dims[rdims[3]]; ++u) {
                        res3 = ReOp<type>::compute(res3, in_data[idx3]);
                        idx3 += in_stride[rdims[3]];
                    }
                    if (k == 0) {
                        res2 = res3;
                    } else {
                        res2 = ReOp<type>::compute(res2, res3);
                    }
                    idx2 += in_stride[rdims[2]];
                }
                if (j == 0) {
                    res1 = res2;
                } else {
                    res1 = ReOp<type>::compute(res1, res2);
                }
                idx1 += in_stride[rdims[1]];
            }
            if (i == 0) {
                res0 = res1;
            } else {
                res0 = ReOp<type>::compute(res0, res1);
            }
            idx0 += in_stride[rdims[0]];
        }
        return res0;
    }
};

template <typename dtype,
        ReduceType type,
        int nDim,
        int rDim>
__global__ void reduce(
        const dtype* src,
        dtype* dst,
        const int* rdim,
        const int* dims,
        const int* i_stride,
        const int* o_stride, int out_size) {
    int reduce_size = 1;
    for (int i = 0; i < rDim; ++i) {
        reduce_size *= dims[rdim[i]];
    }
    float reduce_size_1 = 1.f / ((float)reduce_size);
    int bid = blockIdx.x;

    int out_idx = bid;
    //init;
    int in_idx = IndexCompute<nDim>::input_idx(i_stride, o_stride, out_idx);
    float res = ReduceCompute<rDim, CUDA_NUM_THREADS, type>::compute(
            dims, rdim, i_stride, src, in_idx);
    dst[out_idx] = res;
    if (Reduce_avg == type) {
        dst[out_idx] *= reduce_size_1;
    }
}

__global__
void reduce_unknow(
        const float* src,
        float* dst,
        const int* rdim,
        const int* dims,
        const int* i_stride,
        const int* o_stride, int out_size) {return;}

template <typename dtype,
        ReduceType type,
        int nDim,
        int rDim>
__global__ void reduce_all(
        const dtype* src,
        dtype* dst,
        const int* rdim,
        const int* dims,
        const int* i_stride,
        const int* o_stride,
        int out_size) {

    int reduce_size = 1;
    for (int i = 0; i < rDim; ++i) {
        reduce_size *= dims[rdim[i]];
    }
    float reduce_size_1 = 1.f / ((float)reduce_size);
    //init;
    float res = src[0];
    for (int i = 1; i < reduce_size; ++i) {
        res = ReOp<type>::compute(res, src[i]);
    }
    dst[0] = res;
    if (Reduce_avg == type) {
        dst[0] *= reduce_size_1;
    }
}
}

#define REG_REDUCE_TYPE_KERNEL(REDUCE_TYPE) \
        _kernel_direct_map[REDUCE_TYPE] = { \
        {reduce_unknow}, \
        {reduce_unknow, \
         reduce_all<float, REDUCE_TYPE, 1, 1>}, \
        {reduce_unknow, \
        reduce<float, REDUCE_TYPE, 2, 1>, \
        reduce_all<float, REDUCE_TYPE, 2, 2>}, \
        {reduce_unknow, \
        reduce<float, REDUCE_TYPE, 3, 1>, \
        reduce<float, REDUCE_TYPE, 3, 2>, \
        reduce_all<float, REDUCE_TYPE, 3, 3>}, \
        {reduce_unknow, \
        reduce<float, REDUCE_TYPE, 4, 1>, \
        reduce<float, REDUCE_TYPE, 4, 2>, \
        reduce<float, REDUCE_TYPE, 4, 3>, \
        reduce_all<float, REDUCE_TYPE, 4, 4>}}

template <typename dtype>
void async_copy_to_buffer(Buffer<NV> &buffer,
        dtype* data, unsigned long size, cudaStream_t stream) {
    buffer.re_alloc(size * sizeof(dtype));
    cudaMemcpyAsync(buffer.get_data_mutable(), data,
            size * sizeof(dtype), cudaMemcpyHostToDevice, stream);
}

template <>
SaberStatus SaberReduce<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param, Context<NV>& ctx) {
    this->_ctx = &ctx;

    if (_template_reduction) {
        auto stream = _ctx->get_compute_stream();

        auto i_stride = inputs[0]->get_stride();
        auto o_stride = outputs[0]->get_stride();
        std::vector<int> ndim(inputs[0]->valid_shape());
        async_copy_to_buffer<int>(_rdim_b,
                param.reduce_dim.data(),
                param.reduce_dim.size(), stream);
        async_copy_to_buffer<int>(_ndim_b,
                inputs[0]->valid_shape().data(),
                inputs[0]->valid_shape().size(), stream);
        async_copy_to_buffer<int>(_i_stride_b,
                i_stride.data(), i_stride.size(), stream);
        async_copy_to_buffer<int>(_o_stride_b,
                o_stride.data(), o_stride.size(), stream);
        return SaberSuccess;

    } else {
        return _impl->create(inputs, outputs, param, ctx);
    }
}

template <>
SaberStatus SaberReduce<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    
    if (param.reduce_type == Reduce_avg) {
        _template_reduction = true;
    }

    if (_template_reduction) {
        REG_REDUCE_TYPE_KERNEL(Reduce_avg);
        REG_REDUCE_TYPE_KERNEL(Reduce_min);
        REG_REDUCE_TYPE_KERNEL(Reduce_max);
        REG_REDUCE_TYPE_KERNEL(Reduce_sum);
        REG_REDUCE_TYPE_KERNEL(Reduce_prod);
    } else {
        _impl = new VenderReduce<NV, AK_FLOAT>;
        _impl->init(inputs, outputs, param, ctx);
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberReduce<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param) {

    if (_template_reduction) {
        int out_size = outputs[0]->valid_size();
        _kernel_direct_map[param.reduce_type]
        [inputs[0]->dims()]
        [param.reduce_dim.size()] << < out_size, 1,
            0, _ctx->get_compute_stream() >> > (
                    (const float *) inputs[0]->data(),
                    (float *) outputs[0]->mutable_data(),
                    (const int *) _rdim_b.get_data(),
                    (const int *) _ndim_b.get_data(),
                    (const int *) _i_stride_b.get_data(),
                    (const int *) _o_stride_b.get_data(),
                    outputs[0]->valid_size());
        return SaberSuccess;
    } else {
        return _impl->dispatch(inputs, outputs, param);
    }

}

template class SaberReduce<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberReduce, ReduceParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberReduce, ReduceParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.
