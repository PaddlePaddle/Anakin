#include "saber/lite/funcs/saber_slice.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
SaberStatus SaberSlice<Dtype>::dispatch(const std::vector<Tensor<Dtype> *> &inputs, \
std::vector<Tensor<Dtype> *> &outputs, SliceParam<Tensor<Dtype>> &param) {
    int offset_slice_axis = 0;
    const Dtype* din = inputs[0]->data();
    const int in_slice_axis = inputs[0]->valid_shape()[param.axis];
    for (int i = 0; i < outputs.size(); ++i) {
        Dtype* dout = outputs[i]->mutable_data();
        const int out_slice_axis = outputs[i]->valid_shape()[param.axis];
        for (int n = 0; n < _slice_num; ++n) {
            const int out_offset = n * out_slice_axis * _slice_size;
            const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
            memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(Dtype) * out_slice_axis * _slice_size);
        }
        offset_slice_axis += out_slice_axis;
    }
    return SaberSuccess;
}

template class SaberSlice<float>;

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


