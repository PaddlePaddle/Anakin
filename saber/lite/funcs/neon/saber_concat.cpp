#include "saber/lite/funcs/saber_concat.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberConcat::SaberConcat(const ParamBase *param) {
    _param = (const ConcatParam*)param;
    this->_flag_param = true;
}

SaberConcat::~SaberConcat() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberConcat::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const ConcatParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberConcat::load_param(std::istream &stream, const float *weights) {
    int axis;
    stream >> axis;
    _param = new ConcatParam(axis);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberConcat::load_param(FILE *fp, const float *weights) {
    int axis;
    fscanf(fp, "%d\n", &axis);
    _param = new ConcatParam(axis);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
SaberStatus SaberConcat::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                              std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load concat param first\n");
        return SaberNotInitialized;
    }

    unsigned long input_size = inputs.size();

    Shape shape_out = inputs[0]->valid_shape();

    //! compute output shape
    for (int i = 1; i < input_size; ++i) {
        Shape sh = inputs[i]->valid_shape();
        for (int j = 0; j < sh.dims(); ++j) {
            if (j == _param->_axis) { continue; }
            LCHECK_EQ(shape_out[j], sh[j], "All inputs must have the same shape, except at concat_axis.");
        }
        shape_out[_param->_axis] += sh[_param->_axis];
    }
    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberConcat::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                              std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                              Context &ctx) {
    if (!this->_flag_param) {
        printf("load concat param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;
    _num_concats = inputs[0]->count_valid(0, _param->_axis);
    _concat_input_size = inputs[0]->count_valid(_param->_axis + 1, inputs[0]->dims());
    this->_flag_init = true;
    return SaberSuccess;
}

template <typename dtype>
void concat_kernel_arm(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}


SaberStatus SaberConcat::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                  std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init concat first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    int input_size = inputs.size();

    //! get output data, valid shape and stride shape
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[_param->_axis];

    if (inputs.size() == 1) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    float* dout = outputs[0]->mutable_data();

    for (int i = 0; i < input_size; ++i) {
        Shape sh_in = inputs[i]->valid_shape();
        const float* din = inputs[i]->data();
        const int in_concat_axis = sh_in[_param->_axis];
        for (int n = 0; n < _num_concats; ++n) {
            concat_kernel_arm<float>(in_concat_axis * _concat_input_size,
                            din + n * in_concat_axis * _concat_input_size,
                            dout + (n * out_concat_axis + offset_concat_axis)
                                       * _concat_input_size);
        }
        offset_concat_axis += in_concat_axis;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("concat time: %f\n", ts);
    OpTimer::add_timer("concat", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberConcat);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE
