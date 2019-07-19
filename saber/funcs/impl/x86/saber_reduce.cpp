
#include "saber/funcs/impl/x86/saber_reduce.h"

namespace anakin {
namespace saber {
namespace {

template <ReduceType type>
class ReOp{
public:
    static float compute(float a, float b) {
        return -1.f;
    }
};

template <>
float ReOp<Reduce_unknow>::compute(float a, float b) {
    LOG(FATAL) << "reduce type is not init yet!!!!";
    return 0;
}

template <>
float ReOp<Reduce_max>::compute(float a, float b) {
    return ((a > b) ? a : b);
}

template <>
float ReOp<Reduce_min>::compute(float a, float b) {
    return ((a > b) ? b : a);
}

template <>
float ReOp<Reduce_sum>::compute(float a, float b) {
    return a + b;
}

template <>
float ReOp<Reduce_avg>::compute(float a, float b) {
    return a + b;
}

template <>
float ReOp<Reduce_prod>::compute(float a, float b) {
    return a * b;
}

template <int nDim>
class IndexCompute {
public:
    static int input_idx(const int* dims,
                         const int* odims,
                         int out_idx);
};

template <>
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
int IndexCompute<1>::input_idx(
        const int* in_stride,
        const int* out_stride,
        int out_idx) {

    int i0 = out_idx / out_stride[0];
    int idx = i0 * in_stride[0];
    return idx;
}

template <int rdim, ReduceType type>
class ReduceCompute{
public:
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float* in_data, int in_idx) {
        return 0;
    }
};

template <ReduceType type>
class ReduceCompute<1, type> {
public:
    static float compute(
            const int* dims,
            const int* rdims,
            const int* in_stride,
            const float *in_data, int in_idx) {

        float res = in_data[in_idx];
        int idx = in_idx + in_stride[rdims[0]];
#pragma ivdep
        for (int i = 1; i < dims[rdims[0]]; ++i) {
            res = ReOp<type>::compute(res, in_data[idx]);
            idx += in_stride[rdims[0]];
        }
        return res;
    }
};

template <ReduceType type>
class ReduceCompute<2, type> {
public:
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
#pragma ivdep
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

template <ReduceType type>
class ReduceCompute<3, type> {
public:
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
#pragma ivdep
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

template <ReduceType type>
class ReduceCompute<4, type> {
public:
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
#pragma ivdep
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
void reduce(
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
#pragma omp parallel for
    for (int x = 0; x < out_size; ++x) {
        int out_idx = x;
        //init;
        int in_idx = IndexCompute<nDim>::input_idx(i_stride, o_stride, out_idx);
        float res = ReduceCompute<rDim, type>::compute(
                dims, rdim, i_stride, src, in_idx);
        dst[out_idx] = res;
        if (Reduce_avg == type) {
            dst[out_idx] *= reduce_size_1;
        }
    }
}

void reduce_unknow(
        const float* src,
        float* dst,
        const int* rdim,
        const int* dims,
        const int* i_stride,
        const int* o_stride, int out_size) {
    LOG(FATAL) << "reduce type unkonw!!!";
}

template <typename dtype,
        ReduceType type,
        int nDim,
        int rDim>
void reduce_all(
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
#pragma ivdep
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

template <>
SaberStatus SaberReduce<X86, AK_FLOAT>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ReduceParam<X86>& param, Context<X86>& ctx) {
    return SaberSuccess;
}

template <>
SaberStatus SaberReduce<X86, AK_FLOAT>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ReduceParam<X86>& param, Context<X86>& ctx) {

    REG_REDUCE_TYPE_KERNEL(Reduce_avg);
    REG_REDUCE_TYPE_KERNEL(Reduce_min);
    REG_REDUCE_TYPE_KERNEL(Reduce_max);
    REG_REDUCE_TYPE_KERNEL(Reduce_sum);
    REG_REDUCE_TYPE_KERNEL(Reduce_prod);

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberReduce<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ReduceParam<X86>& param) {

    auto i_stride = inputs[0]->get_stride();
    auto o_stride = outputs[0]->get_stride();
    std::vector<int> ndim;

    for (auto i : inputs[0]->valid_shape()) {
        ndim.push_back(i);
    }
    _kernel_direct_map[param.reduce_type][inputs[0]->dims()][param.reduce_dim.size()](
            (const float*)inputs[0]->data(),
            (float*)outputs[0]->mutable_data(),
            param.reduce_dim.data(), ndim.data(),
            i_stride.data(), o_stride.data(),
            outputs[0]->valid_size());

    return SaberSuccess;
}

template class SaberReduce<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberReduce, ReduceParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberReduce, ReduceParam, X86, AK_INT8);

} // namespace saber.
} // namespace anakin.
