#include "saber/funcs/impl/x86/saber_softmax.h"
#include <cmath>
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


template <DataType OpDtype>
SaberStatus SaberSoftmax<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<X86>& param) {

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    const OpDataType* data_in = (const OpDataType*)inputs[0]->data();
    OpDataType* data_out = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* max_data = (OpDataType*)this->_max_data.mutable_data();
    const int* input_stride = (const int*)_input_stride.data();
    const int* output_stride = (const int*)_output_stride.data();
    int total_num = _inner_num * _outer_num;
    int axis = param.axis;
    Shape sh_in = inputs[0]->valid_shape();
    Shape sh_out = outputs[0]->valid_shape();

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
