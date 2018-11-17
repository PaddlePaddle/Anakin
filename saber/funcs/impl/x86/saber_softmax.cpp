#include "saber/funcs/impl/x86/saber_softmax.h"
#include <cmath>
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSoftmax<X86, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSoftmax<X86, OpDtype>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<X86>& param, Context<X86>& ctx) {
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    CHECK_EQ(shape_in == shape_out, true) << "valid shapes must be the same";
    _outer_num = inputs[0]->count_valid(0, param.axis);
    _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    _axis_size = shape_in[param.axis];
    _max_data.reshape(Shape({1, 1, 1, _axis_size}));
    _dims = shape_in.size();
    Shape sh({1, 1, 1, _dims});
    _input_stride.reshape(sh);
    _output_stride.reshape(sh);
    memcpy(_input_stride.mutable_data(), (inputs[0]->get_stride()).data(), sizeof(int) * _dims);
    memcpy(_output_stride.mutable_data(), (outputs[0]->get_stride()).data(), sizeof(int) * _dims);
    return SaberSuccess;
}
template <typename dtype>
void _max(int n, const dtype *x, dtype *max_data) {
    max_data[0] = x[0];
    for (int c = 1; c < n; ++c) {
        max_data[0] = max_data[0] > x[c] ? max_data[0] : x[c];
    }
}
template <typename dtype>
void _sub(int n, dtype alpha, const dtype *x, dtype *y) {
    for (int c = 0; c < n; ++c) {
        y[c] = x[c] - alpha;
    }
}

template <typename dtype>
void _exp(int n, const dtype *a, dtype *r) {
#if 1
    vsExp(n, a, r);
#else
    #pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        r[c] = expf(a[c]);
    }
#endif
}

template <typename dtype>
void _sum(int n, const dtype *x, dtype *sum_data) {
    sum_data[0] = 0;
    for (int c = 0; c < n; ++c) {
        sum_data[0] += x[c];
    }
}
template <typename dtype>
void _scal (int n, dtype alpha, dtype *x) {
#if 0
    cblas_sscal(n, alpha, x, 1);
#else
#pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        x[c] *= alpha;
    }
#endif
}

template <DataType OpDtype>
SaberStatus SaberSoftmax<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<X86>& param) {

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    int axis = param.axis;
    Shape sh_in = inputs[0]->valid_shape();
    Shape sh_out = outputs[0]->valid_shape();
    bool use_avx2 = true;
    use_avx2 = use_avx2 && (sh_in.count(axis + 1) == 1);
#if defined(__AVX2__) and defined(__FMA__)
    if (use_avx2) {
        int num = sh_in.count(0, axis);
        int channel = sh_in.count(axis);

        const float *src_ptr = (const float *) inputs[0]->data();
        float *dst_ptr = (float *) outputs[0]->mutable_data();
        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

#pragma omp parallel for schedule(static)
        for (int ou = 0; ou < num; ou++) {
            const float *src_data = src_ptr + ou * channel;
            float *dst_data = dst_ptr + ou * channel;
            float scalar = 0;

            _max(channel, src_data, &scalar);
            _sub(channel, scalar, src_data, dst_data);
            _exp(channel, dst_data, dst_data);
            _sum(channel, dst_data, &scalar);
            _scal(channel, float(1.f) / scalar, dst_data);
        }
        return SaberSuccess;
    }
#endif

    const OpDataType* data_in = (const OpDataType*)inputs[0]->data();
    OpDataType* data_out = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* max_data = (OpDataType*)this->_max_data.mutable_data();
    const int* input_stride = (const int*)_input_stride.data();
    const int* output_stride = (const int*)_output_stride.data();
    int total_num = _inner_num * _outer_num;

    #pragma omp parallel for schedule(static)

    for (int num = 0; num < total_num; ++num) {
        int num_tmp = num;
        int in_index = 0, out_index = 0;

        for (int i = _dims - 1; i >= 0; --i) {
            if (i == axis) {
                continue;
            }

            int pos = num_tmp % sh_in[i];
            in_index += pos * input_stride[i];
            out_index += pos * output_stride[i];
            num_tmp /= sh_in[i];
        }

        OpDataType max = std::numeric_limits<OpDataType>::lowest();

        for (int i = 0; i < _axis_size; ++i) {
            max = data_in[in_index] > max ? data_in[in_index] : max;
            in_index += input_stride[axis];
        }

        OpDataType sum = (OpDataType)0;

        for (int i = 0; i < _axis_size; ++i) {
            in_index -= input_stride[axis];
            max_data[_axis_size - i - 1] = expf(data_in[in_index] - max);
            sum += max_data[_axis_size - i - 1];
        }

        for (int i = 0; i < _axis_size; ++i) {
            data_out[out_index] = max_data[i] / sum;
            out_index += output_stride[axis];
        }
    }

    return SaberSuccess;
}
template class SaberSoftmax<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, X86, AK_INT8);
}
} // namespace anakin

