#include "saber/lite/funcs/saber_slice.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberSlice::SaberSlice(ParamBase *param) {
    _param = (SliceParam*)param;
    _slice_points = _param->_points;
    this->_flag_param = true;
}

SaberSlice::~SaberSlice() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberSlice::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (SliceParam*)param;
    _slice_points = _param->_points;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberSlice::load_param(std::istream &stream, const float *weights) {
    int axis;
    int size;
    std::vector<int> points;
    stream >> axis >> size;
    points.resize(size);
    for (int i = 0; i < size; ++i) {
        stream >> points[i];
    }
    _param = new SliceParam(axis, points);
    _slice_points = _param->_points;
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberSlice::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberSlice::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                             std::vector<Tensor<CPU> *> &outputs) {
    if (!this->_flag_param) {
        LOGE("ERROR: load slice param first\n");
        return SaberNotInitialized;
    }

    SaberStatus status;
    //! input size is equal to 1
    Shape shape_in = inputs[0]->valid_shape();
    int top_size = outputs.size();
    int slice_points_size = _slice_points.size();
    int axis_size = shape_in[_param->_axis];
    LCHECK_EQ(bool((top_size > 0) || (slice_points_size > 0)), true, \
        "ERROR: output shapes number is 0 and slice points size is 0");

    if (slice_points_size > 0) {
        LCHECK_EQ(slice_points_size + 1, top_size, "ERROR: error params or ouput size\n");
        int prev = 0;
        Shape sh = shape_in;
        for (int i = 0; i < slice_points_size; ++i) {
            LCHECK_GT(_slice_points[i], prev, "ERROR: later should > prev\n");
            LCHECK_LT(_slice_points[i], axis_size, "ERROR: slice point exceed\n");
            sh[_param->_axis] = _slice_points[i] - prev;
            outputs[i]->set_shape(sh);
            prev = _slice_points[i];
            sh = shape_in;
        }
        LCHECK_GT(axis_size - prev, 0, "ERROR: slice point exceed");
        sh[_param->_axis] = axis_size - prev;
        return outputs[slice_points_size]->set_shape(sh);
    } else {

        LCHECK_EQ(axis_size % top_size, 0, "ERROR: size in slice axis should divide exactly by top size");
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

SaberStatus SaberSlice::init(const std::vector<Tensor<CPU> *> &inputs,
                             std::vector<Tensor<CPU> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        LOGE("ERROR: load slice param first\n");
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
SaberStatus SaberSlice::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                 std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init slice first\n");
        return SaberNotInitialized;
    }

    // bool flag_precision = inputs[0]->get_dtype() == this->get_op_precision();

    // for (int j = 0; j < outputs.size(); ++j) {
    //     flag_precision = flag_precision && (this->get_op_precision() == outputs[j]->get_dtype());
    // }
    // if (!flag_precision) {
    //     LOGE("ERROR: input dtype: %d, output dtype: %d, op precision type: %d must be the same\n", \
    //     inputs[0]->get_dtype(), outputs[0]->get_dtype(), this->get_op_precision());
    //     return SaberInvalidValue;
    // }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif
/*
    DataType op_type = this->get_op_precision();
    if (op_type == AK_FLOAT) {
//        LOGE("fp32 slice\n");
        int offset_slice_axis = 0;
        const float* din = static_cast<const float*>(inputs[0]->data());
        const int in_slice_axis = inputs[0]->valid_shape()[_param->_axis];
        for (int i = 0; i < outputs.size(); ++i) {
            float* dout = static_cast<float*>(outputs[i]->mutable_data());
            const int out_slice_axis = outputs[i]->valid_shape()[_param->_axis];
            for (int n = 0; n < _slice_num; ++n) {
                const int out_offset = n * out_slice_axis * _slice_size;
                const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
                memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(float) * out_slice_axis * _slice_size);
            }
            offset_slice_axis += out_slice_axis;
        }
    } else if (op_type == AK_INT8) {
//        LOGE("int8 slice\n");
        int offset_slice_axis = 0;
        const char* din = static_cast<const char*>(inputs[0]->data());
        const int in_slice_axis = inputs[0]->valid_shape()[_param->_axis];
        for (int i = 0; i < outputs.size(); ++i) {
            char* dout = static_cast<char*>(outputs[i]->mutable_data());
            const int out_slice_axis = outputs[i]->valid_shape()[_param->_axis];
            for (int n = 0; n < _slice_num; ++n) {
                const int out_offset = n * out_slice_axis * _slice_size;
                const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
                memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(char) * out_slice_axis * _slice_size);
            }
            offset_slice_axis += out_slice_axis;
        }
    } else {
        LOGE("ERROR: slice unsupported precision type: %d\n", this->get_op_precision());
        return SaberUnImplError;
    }
*/

    const void* din = nullptr;
    std::vector<void*> dout(outputs.size());

    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType op_type = this->get_op_precision();
    if (op_type == AK_INT8) {
        if (inputs[0]->get_dtype() != AK_INT8) {
           LOGI("slice int8 trans input, fp32 to int8\n");
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_fp32_to_int8(*inputs[0], _tmp_in, _ctx);
            din = _tmp_in.data();
        } else {
           LOGI("slice int8 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else if (this->get_op_precision() == AK_FLOAT) {
        if (inputs[0]->get_dtype() != AK_FLOAT) {
           LOGI("slice fp32 trans input, int8 to fp32\n");
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_int8_to_fp32(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], _ctx);
            din = _tmp_in.data();
        } else {
           LOGI("slice fp32 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else {
        LOGE("ERROR: unsupported input data type!!\n");
        return SaberInvalidValue;
    }

    DataType tensor_out_type = outputs[0]->get_dtype();
    if (op_type == AK_INT8 && tensor_out_type == AK_INT8) {
        LOGI("slice int8 trans output, no trans\n");
        for (int i = 0; i < outputs.size(); i++){
            dout[i] = outputs[i]->mutable_data();
        }
    } else if (op_type == AK_FLOAT && tensor_out_type == AK_FLOAT) {
        LOGI("slice fp32 trans output, no trans\n");
        for (int i = 0; i < outputs.size(); i++){
            dout[i] = outputs[i]->mutable_data();
        }
    }else if (op_type == AK_INT8 && tensor_out_type == AK_FLOAT){
        LOGI("slice fp32 trans output, trans int8 to fp32 \n");
        for (int i = 0; i < outputs.size(); i++){
            _tmp_out[i].set_dtype(AK_FLOAT);
            _tmp_out[i].reshape(outputs[i]->valid_shape());
            dout[i] = _tmp_out[i].mutable_data();
        }
    } else if (op_type == AK_FLOAT && tensor_out_type == AK_INT8){
        LOGI("slice int8 trans output, trans fp32 to int8 \n");
        // dout = _tmp_out.mutable_data();
        for (int i = 0; i < outputs.size(); i++){
            _tmp_out[i].set_dtype(AK_FLOAT);
            _tmp_out[i].reshape(outputs[i]->valid_shape());
            dout[i] = _tmp_out[i].mutable_data();
        }
    }  else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }

    if (op_type == AK_FLOAT) {
        //! do nothing
        if (outputs[0]->get_dtype() != AK_FLOAT) {
            LOGE("ERROR: unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }

    if (op_type == AK_FLOAT) {
       LOGE("fp32 slice\n");
        int offset_slice_axis = 0;
        const float* dina = static_cast<const float*>(din);
        const int in_slice_axis = inputs[0]->valid_shape()[_param->_axis];
        for (int i = 0; i < outputs.size(); ++i) {
            float* doutb = static_cast<float*>(dout[i]);
            const int out_slice_axis = outputs[i]->valid_shape()[_param->_axis];
            for (int n = 0; n < _slice_num; ++n) {
                const int out_offset = n * out_slice_axis * _slice_size;
                const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
                memcpy((void*)(doutb + out_offset), (void*)(dina + in_offset), \
                sizeof(float) * out_slice_axis * _slice_size);
            }
            offset_slice_axis += out_slice_axis;
        }
    } else if (op_type == AK_INT8) {
       LOGE("int8 slice\n");
        int offset_slice_axis = 0;
        const char* dina = static_cast<const char*>(din);
        const int in_slice_axis = inputs[0]->valid_shape()[_param->_axis];
        for (int i = 0; i < outputs.size(); ++i) {
            char* doutb = static_cast<char*>(dout[i]);
            const int out_slice_axis = outputs[i]->valid_shape()[_param->_axis];
            for (int n = 0; n < _slice_num; ++n) {
                const int out_offset = n * out_slice_axis * _slice_size;
                const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
                memcpy((void*)(doutb + out_offset), (void*)(dina + in_offset), \
                sizeof(char) * out_slice_axis * _slice_size);
            }
            offset_slice_axis += out_slice_axis;
        }
    } else {
        LOGE("ERROR: slice unsupported precision type: %d\n", this->get_op_precision());
        return SaberUnImplError;
    }

    if (op_type == AK_INT8) {
        if (tensor_out_type == AK_FLOAT) {
           LOGI("slice trans output_final, int8 to fp32\n");
           for (int i = 0; i < outputs.size(); ++i) {
                trans_tensor_int8_to_fp32(_tmp_out[i], *outputs[i], outputs[i]->get_scale()[0], _ctx);
            }
        }
    }

    if (op_type == AK_FLOAT) {
        if (tensor_out_type == AK_INT8) {
           LOGI("slice trans output_final, fp32 to int8\n");
            // trans_tensor_fp32_to_int8(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], _ctx);
           for (int i = 0; i < outputs.size(); ++i) {
               trans_tensor_fp32_to_int8(_tmp_out[i], *outputs[i], _ctx);
            }
        }
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    LOGI("slice time %s: %f\n", this->_op_name.c_str(), ts);
    GOPS ops;
    // fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("slice", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberSlice);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


