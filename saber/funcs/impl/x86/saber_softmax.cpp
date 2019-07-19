#include "saber/funcs/impl/x86/saber_softmax.h"
#include <cmath>
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSoftmax<X86, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    if (inputs[0]->get_dtype() != AK_FLOAT) {
        _input_scale.re_alloc(inputs[0]->valid_shape(), AK_FLOAT);
    }
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

    if (inputs[0]->get_dtype() != AK_FLOAT) {
        utils::try_expand_tensor(_input_scale, inputs[0]->valid_shape());
    }
    return SaberSuccess;
}

template <typename dtype>
void _max(int n, const dtype* x, dtype* output_max_data) {
//    print_vec(x,n,"max");
    dtype max_data = x[0];
    for (int c = 1; c < n; ++c) {
        max_data = max_data > x[c] ? max_data : x[c];
    }

    output_max_data[0] = max_data;
}
template <typename dtype>
void _sub(int n, dtype alpha, const dtype* x, dtype* y) {
    for (int c = 0; c < n; ++c) {
        y[c] = x[c] - alpha;
    }
}

template <typename dtype>
void _exp(int n, const dtype* a, dtype* r) {
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
void _sum(int n, const dtype* x, dtype* sum_data) {
    dtype sum = 0;
    for (int c = 0; c < n; ++c) {
        sum += x[c];
    }

    sum_data[0] = sum;
}
template <typename dtype>
void _scal(int n, dtype alpha, dtype* x) {
#if 0
    cblas_sscal(n, alpha, x, 1);
#else
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

    if (sh_in.get_layout() == Layout_NHWC) {
        sh_in = Shape({sh_in.num(), sh_in.channel(), sh_in.height(), sh_in.width()});
    }

    int axis_size = sh_in[axis];
    int outer_dim = sh_in.count(0, param.axis);
    int inner_dim = sh_in.count(param.axis + 1, inputs[0]->dims());
    int batch_size = outer_dim * inner_dim;
    const float* src_ptr = nullptr;
    float* dst_ptr = (float*) outputs[0]->mutable_data();
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        src_ptr = static_cast<const float*>(inputs[0]->data());
    } else if (inputs[0]->get_dtype() == AK_UINT8) {
                DLOG(INFO) << "dispatch convert uint8 fp32";
        utils::ScaleUtils::scale_uint8_fp32(_input_scale, *inputs[0]);
        src_ptr = static_cast<const float*>(_input_scale.data());
    }else{
        LOG(INFO)<<"not support input "<<inputs[0]->get_dtype();
    }
    if (avx2_can_used()){
#if defined(__AVX2__) and defined(__FMA__)
#pragma omp parallel for schedule(static) if(outer_dim>1)
        for(int outer_id=0; outer_id<outer_dim; outer_id++){
            const float* src_data_outer = src_ptr + outer_id * axis_size*inner_dim;
            float* dst_data_outer = dst_ptr + outer_id * axis_size*inner_dim;
            if (inner_dim == 1){
                avx2_vector_softmax(src_data_outer, axis_size, dst_data_outer);
            }else{
                avx2_vector_softmax_stride(src_data_outer, inner_dim, axis_size, dst_data_outer);
            }
        }
#endif
    } else {
        const OpDataType *data_in = (const OpDataType *) inputs[0]->data();
        OpDataType *data_out = (OpDataType *) outputs[0]->mutable_data();
        OpDataType *max_data = (OpDataType *) this->_max_data.mutable_data();
        const int *input_stride = (const int *) _input_stride.data();
        const int *output_stride = (const int *) _output_stride.data();
        int total_num = _inner_num * _outer_num;

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

            OpDataType sum = (OpDataType) 0;

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
    }


    return SaberSuccess;
}
template class SaberSoftmax<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, X86, AK_INT8);
}
} // namespace anakin

