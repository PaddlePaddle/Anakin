#include "saber/lite/funcs/saber_reshape.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberReshape::SaberReshape(const ParamBase *param) {
    _param = (ReshapeParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberReshape::load_param(const ParamBase *param) {
    _param = (ReshapeParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberReshape::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load concat param first\n");
        return SaberNotInitialized;
    }

    Shape output_shape;
    output_shape.resize(_param->_shape_params.size());
    Shape input_shape = inputs[0]->valid_shape();
    int valid_size = inputs[0]->valid_size();
    int infer_axis = -1;
    int count_axis = 1;
    for (int i = 0; i < _param->_shape_params.size(); ++i) {
        if (_param->_shape_params[i] == 0){
            LCHECK_LT(i, input_shape.size(), "wrong parameters, exceed input dims");
            output_shape[i] = input_shape[i];
            count_axis *= input_shape[i];
        } else if (_param->_shape_params[i] > 0){
            output_shape[i] = _param->_shape_params[i];
            count_axis *= _param->_shape_params[i];
        } else {
            output_shape[i] = -1;
            infer_axis = i;
        }
    }

    if (infer_axis >= 0){
        output_shape[infer_axis] = valid_size / count_axis;
    }
    return outputs[0]->set_shape(output_shape);
}

SaberStatus SaberReshape::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                               Context &ctx) {
    if (!this->_flag_param) {
        printf("load concat param first\n");
        return SaberNotInitialized;
    }
    //outputs[0]->share_from(*inputs[0]);
    return SaberSuccess;
}

SaberStatus SaberReshape::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    return SaberSuccess;
}

REGISTER_LAYER_CLASS(SaberReshape);
} //namespace lite

} //namespace saber

} //namespace anakin