#include "saber/lite/funcs/saber_split.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberStatus SaberSplit::load_param(const ParamBase *param) {
    return SaberSuccess;
}

SaberStatus SaberSplit::load_param(std::istream &stream, const float *weights) {
    return SaberSuccess;
}
#if 0
SaberStatus SaberSplit::load_param(FILE *fp, const float* weights) {
    fscanf(fp, "\n");
    return SaberSuccess;
}
#endif
SaberStatus SaberSplit::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i]->set_shape(inputs[0]->valid_shape());
        outputs[i]->share_from(*inputs[0]);
    }
    return SaberSuccess;
}

SaberStatus SaberSplit::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                             std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                             Context &ctx) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i]->share_from(*inputs[0]);
    }
    return SaberSuccess;
}

SaberStatus SaberSplit::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    return SaberSuccess;
}

REGISTER_LAYER_CLASS(SaberSplit);
} //namespace lite

} //namespace saber

} //namespace anakin