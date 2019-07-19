#include "saber/funcs/impl/x86/saber_slice_v2.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
SaberStatus SaberSliceV2<X86, OpDtype>::create(const std::vector<Tensor<X86>*>& inputs,
                    std::vector<Tensor<X86>*>& outputs,
                    SliceV2Param<X86> &param,
                    Context<X86> &ctx) {
    auto starts = param.starts;
    auto ends = param.ends;
    auto axes = param.axes;
    CHECK_EQ(axes.size(), starts.size()) << "the size of axes and starts are not equal ";
    CHECK_EQ(ends.size(), starts.size()) << "the size of starts and ends are not valid";
    _starts.resize(starts.size());
    _ends.resize(ends.size());
    Shape output_shape = inputs[0]->valid_shape();
    Shape input_shape = inputs[0]->valid_shape();
    for (int i = 0; i < starts.size(); i++) {
        int dim_value = input_shape[axes[i]];
        int start = starts[i] < 0 ? starts[i] + dim_value : starts[i];
        int end = ends[i] < 0 ? ends[i] + dim_value : ends[i];
        start = std::max(start, 0);
        start = std::min(start, dim_value);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        output_shape[axes[i]] = end - start;
        _starts[i] = start;
        _ends[i] = end;
    }
    return SaberSuccess;
}


template <DataType OpDtype>
SaberStatus SaberSliceV2<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    SliceV2Param<X86>& param) {

    //! inputs only has one tensor
    Shape shape_in = inputs[0]->valid_shape();
    auto axes = param.axes;
    CHECK_EQ(outputs.size(), 1) << "SliceV2 only support one output"; 
    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    auto out_stride = outputs[0]->get_stride();
    auto in_stride = inputs[0]->get_stride();
    int inner = inputs[0]->count_valid(param.axes.back() + 1, inputs[0]->dims());
    int out_outer_stride = outputs[0]->count_valid(param.axes[0], inputs[0]->dims());
    int in_outer_stride = inputs[0]->count_valid(param.axes[0], inputs[0]->dims());
    int count = outputs[0]->valid_size();
    auto out_shape = outputs[0]->valid_shape();

    for (int i = 0; i < count; i++) {
        int out_id = i / out_outer_stride;
        int inner_id = i % inner;
        int new_i = i / inner;
        int in_offset = inner_id + out_id * in_outer_stride;
        for (int k = _starts.size() - 1; k >= 0; k--) {
            int cur_id = new_i % out_shape[axes[k]];
            in_offset += (cur_id + _starts[k]) * in_stride[axes[k]];
            new_i /= out_shape[axes[k]];
        } 
        out_data[i] = in_data[in_offset];
    }

    return SaberSuccess;

}
DEFINE_OP_TEMPLATE(SaberSliceV2, SliceV2Param, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSliceV2, SliceV2Param, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
