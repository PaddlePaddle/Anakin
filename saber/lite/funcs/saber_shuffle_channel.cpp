#include "saber/lite/funcs/saber_shuffle_channel.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{
template <typename Dtype>
void shuffle_kernel(Dtype* output, const Dtype* input, int group_row, int group_col, int len) {
    for (int i = 0; i < group_row; ++i) {
        for (int j = 0; j < group_col; ++j) {
            const Dtype* p_i = input + (i * group_col + j) * len;
            Dtype* p_o = output + (j * group_row + i ) * len;
            memcpy(p_o, p_i, len * sizeof(Dtype));
        }
    }
}

SaberShuffleChannel::SaberShuffleChannel(ParamBase *param) {
    _param = (ShuffleChannelParam*)param;
    this->_flag_param = true;
}

SaberShuffleChannel::~SaberShuffleChannel() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberShuffleChannel::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (ShuffleChannelParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberShuffleChannel::load_param(std::istream &stream, const float *weights) {
    int group;
    stream >> group;
    _param = new ShuffleChannelParam(group);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberShuffleChannel::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberShuffleChannel::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                             std::vector<Tensor<CPU> *> &outputs) {
    if (!this->_flag_param) {
        LOGE("ERROR: load shuffle channel param first\n");
        return SaberNotInitialized;
    }
    if (inputs[0]->channel() % _param->_group != 0) {
        LOGE("ERROR: shuffle channel can not be divided by group\n");
        return SaberInvalidValue;
    }
    return outputs[0]->set_shape(inputs[0]->valid_shape());
}

SaberStatus SaberShuffleChannel::init(const std::vector<Tensor<CPU> *> &inputs,
                             std::vector<Tensor<CPU> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        LOGE("ERROR: load shuffle channel param first\n");
        return SaberNotInitialized;
    }
    // get context
    this->_ctx = &ctx;
    //outputs[0]->share_from(*inputs[0]);
    this->_flag_init = true;
    return SaberSuccess;
}


//template <typename Dtype>
SaberStatus SaberShuffleChannel::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                 std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init shuffle channel first\n");
        return SaberNotInitialized;
    }

    bool flag_precision = inputs[0]->get_dtype() == outputs[0]->get_dtype();
    flag_precision = flag_precision && (inputs[0]->get_dtype() == this->get_op_precision());
    // if (!flag_precision) {
    //     LOGE("ERROR: input dtype: %d, output dtype: %d, op precision type: %d must be the same\n", \
    //         inputs[0]->get_dtype(), outputs[0]->get_dtype(), this->get_op_precision());
    //     return SaberInvalidValue;
    // }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int fea_size = channel * height * width;
    int spatial_size = height * width;

    int group_row = _param->_group;
    int group_col = channel / _param->_group;

    const void* din = nullptr;
    void* dout = nullptr;

    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType op_type = this->get_op_precision();
    if (op_type == AK_INT8) {
        if (inputs[0]->get_dtype() != AK_INT8) {
           LOGI("shuffle_channel int8 trans input, fp32 to int8\n");
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_fp32_to_int8(*inputs[0], _tmp_in, _ctx);
            din = _tmp_in.data();
        } else {
           LOGI("shuffle_channel int8 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else if (this->get_op_precision() == AK_FLOAT) {
        if (inputs[0]->get_dtype() != AK_FLOAT) {
           LOGI("shuffle_channel fp32 trans input, int8 to fp32\n");
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_int8_to_fp32(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], _ctx);
            din = _tmp_in.data();
        } else {
           LOGI("shuffle_channel fp32 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else {
        LOGE("ERROR: unsupported input data type!!\n");
        return SaberInvalidValue;
    }

    DataType tensor_out_type = outputs[0]->get_dtype();
    if (op_type == AK_INT8 && tensor_out_type == AK_INT8) {
        LOGI("shuffle_channel int8 trans output, no trans\n");
        dout = outputs[0]->mutable_data();
    } else if (op_type == AK_FLOAT && tensor_out_type == AK_FLOAT) {
        LOGI("shuffle_channel fp32 trans output, no trans\n");
        dout = outputs[0]->mutable_data();
    }else if (op_type == AK_INT8 && tensor_out_type == AK_FLOAT){
        LOGI("shuffle_channel fp32 trans output, trans int8 to fp32 \n");
        _tmp_out.set_dtype(AK_FLOAT);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = _tmp_out.mutable_data();
    } else if (op_type == AK_FLOAT && tensor_out_type == AK_INT8){
        LOGI("shuffle_channel int8 trans output, trans fp32 to int8 \n");
        _tmp_out.set_dtype(AK_FLOAT);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = _tmp_out.mutable_data();
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

    if (this->get_op_precision() == AK_FLOAT) {
//        LOGE("fp32 shuffle\n");
        const float* dina = static_cast<const float*>(din);
        float* doutb = static_cast<float*>(dout);
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(doutb + i * fea_size, dina + i * fea_size, group_row, group_col, spatial_size);
        }
    } else if(this->get_op_precision() == AK_INT8) {
//        LOGE("int8 shuffle\n");
        const char* dina = static_cast<const char*>(din);
        char* doutb = static_cast<char*>(dout);
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(doutb + i * fea_size, dina + i * fea_size, group_row, group_col, spatial_size);
        }
    } else {
        LOGE("ERROR: shuffle channel unsupported precision type: %d\n", this->get_op_precision());
        return SaberUnImplError;
    }

    if (op_type == AK_INT8) {
        if (tensor_out_type == AK_FLOAT) {
            LOGI("shuffle_channel trans output_final, fp32 to int8\n");
            trans_tensor_int8_to_fp32(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], _ctx);
        }
    }
    if (op_type == AK_FLOAT) {
        if (tensor_out_type == AK_INT8) {
           LOGI("shuffle_channel trans output_final, fp32 to int8\n");
            // trans_tensor_fp32_to_int8(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], _ctx);
           trans_tensor_fp32_to_int8(_tmp_out, *outputs[0], _ctx);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    LOGI("shuffle time %s: %f\n", this->_op_name.c_str(), ts);
    GOPS ops;
    // fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("shuffle", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberShuffleChannel);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


