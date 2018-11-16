#include "saber/lite/funcs/saber_flatten.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberFlatten::SaberFlatten(ParamBase *param) {
    _param = (const FlattenParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberFlatten::load_param(ParamBase *param) {
    _param = (const FlattenParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberFlatten::load_param(std::istream &stream, const float *weights) {
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberFlatten::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberFlatten::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                             std::vector<Tensor<CPU> *> &outputs) {
    if (!this->_flag_param) {
        //printf("load flatten param first\n");
        LOGE("load flatten param first\n");
        return SaberNotInitialized;
    }

    SaberStatus status;
    //! input size is equal to 1
    Shape shape_in;
    shape_in.resize(2);
    shape_in[0] = inputs[0]->num();
    shape_in[1] = inputs[0]->valid_size() / inputs[0]->num();
    return outputs[0]->set_shape(shape_in);
}

SaberStatus SaberFlatten::init(const std::vector<Tensor<CPU> *> &inputs,
                             std::vector<Tensor<CPU> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        //printf("load flatten param first\n");
        LOGE("load flatten param first\n");
        return SaberNotInitialized;
    }
    // get context
    this->_ctx = &ctx;
    outputs[0]->set_dtype(inputs[0]->get_dtype());
    //outputs[0]->share_from(*inputs[0]);
    this->_flag_init = true;
    return SaberSuccess;
}


//template <typename Dtype>
SaberStatus SaberFlatten::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                 std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_init) {
        //printf("init flatten first\n");
        LOGE("init flatten first\n");
        return SaberNotInitialized;
    }
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberFlatten);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


