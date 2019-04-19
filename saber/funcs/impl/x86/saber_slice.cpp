#include "saber/funcs/impl/x86/saber_slice.h"

namespace anakin{

namespace saber{


template <DataType OpDtype>
SaberStatus SaberSlice<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    SliceParam<X86>& param) {

    //! inputs only has one tensor
    Shape shape_in = inputs[0]->valid_shape();

    int output_size = outputs.size();

    if (output_size == 1) {
        outputs[0]->share_from(*inputs[0]);
        return SaberSuccess;
    }

    int offset_slice_axis = 0;
    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const int in_slice_axis_size = shape_in[param.axis];
    for (int i = 0; i < output_size; ++i) {
        OpDataType* out_data = (OpDataType*)outputs[i]->mutable_data();
        const int out_slice_axis_size = outputs[i]->valid_shape()[param.axis];
        const int out_slice_size = out_slice_axis_size * _slice_size;
        const int slice_count = out_slice_size * _slice_num;
#pragma omp parallel for schedule(static)
        for(int j = 0; j < slice_count; ++j){
            const int _num_slice = j / out_slice_size;
            const int _slice_index = j % out_slice_size;
            const int in_index = _slice_index + (_num_slice * in_slice_axis_size + offset_slice_axis) * _slice_size;
            out_data[j] = in_data[in_index];
        }
        offset_slice_axis += out_slice_axis_size;
    }
    return SaberSuccess;

}
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
