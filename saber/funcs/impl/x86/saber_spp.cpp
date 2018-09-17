#include "saber/funcs/impl/x86/saber_spp.h"
#include "saber/core/tensor_op.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberSpp<X86, OpDtype>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    SPPParam<X86>& param) {

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int spatial_size = outputs[0]->width() * outputs[0]->height();
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        std::vector<OpTensor*> pool_outputs;
        pool_outputs.resize(1);
        for (int i = 0; i < param.pyramid_height; i++) {
            pool_outputs[0] = _pooling_output[i];
            (*_pooling[i])(inputs, pool_outputs, _pooling_param[i], *(this->_ctx));
            int valid_size  = pool_outputs[0]->valid_size();
            int offset = (pow(4, i) - 1) / 3;
            int spatial_size_out = pool_outputs[0]->width() * pool_outputs[0]->height();
            OutDataType* in_data = (OutDataType*)pool_outputs[0]->mutable_data();
            for (int i = 0; i < valid_size; ++i){
                int idx = i / spatial_size_out;
                int out_index = idx * spatial_size + i % spatial_size_out;
                out_data[out_index + offset] = in_data[i];
            }
        }
    }

    return SaberSuccess;
}
} //namespace saber

} //namespace anakin
