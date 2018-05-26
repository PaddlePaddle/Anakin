#include "saber/funcs/impl/impl_define.h"
#include "saber/funcs/impl/cuda/saber_multiclass_nms.h"
#include "saber/funcs/impl/detection_helper.h"
namespace anakin {

namespace saber {

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberMultiClassNMS<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> ::dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          MultiClassNMSParam<OpTensor>& param) {

    cudaStream_t stream = this->_ctx.get_compute_stream();

    DataTensor_in* t_loc = inputs[0];
    DataTensor_in* t_conf = inputs[1];
    int class_num = t_conf->valid_shape()[1];
    const int num = t_loc->num();

    const InDataType* loc_data = t_loc->data();
    const InDataType* conf_data = t_conf->data();


    CUDA_CHECK(cudaMemcpyAsync(_bbox_cpu_data, loc_data, \
                               t_loc->valid_size() * sizeof(InDataType), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(_conf_cpu_data, conf_data, \
                               t_conf->valid_size() * sizeof(InDataType), cudaMemcpyDeviceToHost, stream));

    std::vector<InDataType> result;

    nms_detect(_bbox_cpu_data, _conf_cpu_data, result, num, class_num, _num_priors, param.background_id,
               \
               param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, true);

    if (result.size() == 0) {
        result.resize(7);

        for (int i = 0; i < 7; ++i) {
            result[i] = (InDataType) - 1;
        }

        outputs[0]->reshape({1, 7});
    } else {
        outputs[0]->reshape({result.size() / 7, 7});
    }

    CUDA_CHECK(cudaMemcpyAsync(outputs[0]->mutable_data(), result.data(), \
                               result.size() * sizeof(InDataType), cudaMemcpyHostToDevice, stream));

    return SaberSuccess;
}
template class SaberMultiClassNMS<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NHW, NHW, NW>;
template class SaberMultiClassNMS<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace anakin

} //namespace anakin
