#include "saber/lite/funcs/saber_slice.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberSlice::SaberSlice(const ParamBase *param) {
    _param = (const SliceParam*)param;
    _slice_points = _param->_points;
    this->_flag_param = true;
}

SaberStatus SaberSlice::load_param(const ParamBase *param) {
    _param = (const SliceParam*)param;
    _slice_points = _param->_points;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberSlice::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load slice param first\n");
        return SaberNotInitialized;
    }

    SaberStatus status;
    //! input size is equal to 1
    Shape shape_in = inputs[0]->valid_shape();
    int top_size = outputs.size();
    int slice_points_size = _slice_points.size();
    int axis_size = shape_in[_param->_axis];

    LCHECK_EQ(top_size > 0 || slice_points_size > 0, true, "output shapes number is 0 and slice points size is 0");

    if (slice_points_size > 0) {
        LCHECK_EQ(slice_points_size + 1, top_size, "error params or ouput size");
        int prev = 0;
        Shape sh = shape_in;
        for (int i = 0; i < slice_points_size; ++i) {
            LCHECK_GT(_slice_points[i], prev, " later should > prev");
            LCHECK_LT(_slice_points[i], axis_size, "slice point exceed");
            sh[_param->_axis] = _slice_points[i] - prev;
            outputs[i]->set_shape(sh);
            prev = _slice_points[i];
            sh = shape_in;
        }
        LCHECK_GT(axis_size - prev, 0, "slice point exceed");
        sh[_param->_axis] = axis_size - prev;
        return outputs[slice_points_size]->set_shape(sh);
    } else {

        LCHECK_EQ(axis_size % top_size, 0, "size in slice axis should divide exactly by top size");
        int step = axis_size / top_size;
        Shape sh = shape_in;
        sh[_param->_axis] = step;
        outputs[0]->set_shape(sh);
        for (int i = 1; i < top_size; ++i) {
            _slice_points[i - 1] = i * step;
            status = outputs[i]->set_shape(sh);
            if (status != SaberSuccess) {
                return status;
            }
        }
    }
    return SaberSuccess;
}

SaberStatus SaberSlice::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load slice param first\n");
        return SaberNotInitialized;
    }
    // get context
    this->_ctx = &ctx;
    _slice_num = inputs[0]->count_valid(0, _param->_axis);
    _slice_size = inputs[0]->count_valid(_param->_axis + 1, inputs[0]->dims());
    this->_flag_init = true;
    return SaberSuccess;
}


//template <typename Dtype>
SaberStatus SaberSlice::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init slice first\n");
        return SaberNotInitialized;
    }

    int offset_slice_axis = 0;
    const float* din = inputs[0]->data();
    const int in_slice_axis = inputs[0]->valid_shape()[_param->_axis];
    for (int i = 0; i < outputs.size(); ++i) {
        float* dout = outputs[i]->mutable_data();
        const int out_slice_axis = outputs[i]->valid_shape()[_param->_axis];
        for (int n = 0; n < _slice_num; ++n) {
            const int out_offset = n * out_slice_axis * _slice_size;
            const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
            memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(float) * out_slice_axis * _slice_size);
        }
        offset_slice_axis += out_slice_axis;
    }
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


