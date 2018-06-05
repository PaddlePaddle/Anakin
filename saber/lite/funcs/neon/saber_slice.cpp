#include "saber/lite/funcs/saber_slice.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
SaberStatus SaberSlice<Dtype>::compute_output_shape(const std::vector<Tensor<Dtype> *> &inputs,
                                                    std::vector<Tensor<Dtype> *> &outputs,
                                                    SliceParam<Tensor<Dtype>> &param) {
    SaberStatus status;
    //! input size is equal to 1
    Shape shape_in = inputs[0]->valid_shape();
    int top_size = outputs.size();
    int slice_points_size = param.slice_points.size();
    int axis_size = shape_in[param.axis];

    CHECK_EQ(top_size > 0 || slice_points_size > 0, true) << \
            "output shapes number is 0 and slice points size is 0";

    if (slice_points_size > 0) {
        CHECK_EQ(slice_points_size + 1, top_size) << "error params or ouput size";
        int prev = 0;
        Shape sh = shape_in;
        for (int i = 0; i < slice_points_size; ++i) {
            CHECK_GT(param.slice_points[i], prev) << " later should > prev";
            CHECK_LT(param.slice_points[i], axis_size) << "slice point exceed";
            sh[param.axis] = param.slice_points[i] - prev;
            outputs[i]->set_shape(sh);
            prev = param.slice_points[i];
            sh = shape_in;
        }
        CHECK_GT(axis_size - prev, 0) << "slice point exceed";
        sh[param.axis] = axis_size - prev;
        return outputs[slice_points_size]->set_shape(sh);
    } else {

        CHECK_EQ(axis_size % top_size, 0) << \
                "size in slice axis should divide exactly by top size";
        int step = axis_size / top_size;
        Shape sh = shape_in;
        sh[param.axis] = step;
        outputs[0]->set_shape(sh);
        for (int i = 1; i < top_size; ++i) {
            param.slice_points[i - 1] = i * step;
            status = outputs[i]->set_shape(sh);
            if (status != SaberSuccess) {
                return status;
            }
        }
    }
    return SaberSuccess;
}

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


