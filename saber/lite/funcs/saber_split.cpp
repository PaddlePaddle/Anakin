#include "saber/lite/funcs/saber_split.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberStatus SaberSplit::load_param(ParamBase *param) {
    return SaberSuccess;
}

SaberStatus SaberSplit::load_param(std::istream &stream, const float *weights) {
    return SaberSuccess;
}

SaberStatus SaberSplit::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberSplit::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                             std::vector<Tensor<CPU> *> &outputs) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i]->set_shape(inputs[0]->valid_shape());
        outputs[i]->share_from(*inputs[0]);
    }
    return SaberSuccess;
}

SaberStatus SaberSplit::init(const std::vector<Tensor<CPU> *> &inputs,
                             std::vector<Tensor<CPU> *> &outputs,
                             Context &ctx) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i]->set_dtype(inputs[0]->get_dtype());
        outputs[i]->share_from(*inputs[0]);
    }
    return SaberSuccess;
}

SaberStatus SaberSplit::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                 std::vector<Tensor<CPU> *> &outputs) {
    return SaberSuccess;
}

REGISTER_LAYER_CLASS(SaberSplit);
} //namespace lite

} //namespace saber

} //namespace anakin