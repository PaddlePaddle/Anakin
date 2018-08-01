#include "saber/lite/funcs/saber_flatten.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberFlatten::SaberFlatten(const ParamBase *param) {
    _param = (const FlattenParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberFlatten::load_param(const ParamBase *param) {
    _param = (const FlattenParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberFlatten::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load flatten param first\n");
        return SaberNotInitialized;
    }

    SaberStatus status;
    //! input size is equal to 1
    Shape shape_in = inputs[0]->valid_shape();
    LCHECK_EQ(shape_in.dims(), 4, "only support 4d(NCHW) layout");
    shape_in[1] = inputs[0]->valid_size() / inputs[0]->num();
    shape_in[2] = 1;
    shape_in[3] = 1;
    return outputs[0]->set_shape(shape_in);
}

SaberStatus SaberFlatten::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load slice param first\n");
        return SaberNotInitialized;
    }
    // get context
    this->_ctx = &ctx;
    return SaberSuccess;
}


//template <typename Dtype>
SaberStatus SaberFlatten::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init slice first\n");
        return SaberNotInitialized;
    }
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


