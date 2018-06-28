#include "saber/funcs/impl/arm/saber_concat.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <typename dtype>
void concat_kernel_arm(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
SaberStatus SaberConcat<ARM, OpDtype, inDtype, outDtype, \
LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
        const std::vector<DataTensor_in *>& inputs,
        std::vector<DataTensor_out *>& outputs,
        ConcatParam<OpTensor> &param) {

    int input_size = inputs.size();

    //! get output data, valid shape and stride shape
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    /*
    bool out_cont_flag = outputs[0]->is_continue_mem();
    bool in_cont_flag = inputs[0]->is_continue_mem();
    for (int i = 1; i < input_size; ++i) {
        in_cont_flag &= inputs[i]->is_continue_mem();
    }
    if (!in_cont_flag) {
        LOG(ERROR) << "dis-continued memory is not support yet";
        return SaberUnImplError;
    }
     */

    if (inputs.size() == 1) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    OutDataType* dout = outputs[0]->mutable_data();

    for (int i = 0; i < input_size; ++i) {
        Shape sh_in = inputs[i]->valid_shape();
        const InDataType* din = inputs[i]->data();
        const int in_concat_axis = sh_in[param.axis];
        for (int n = 0; n < _num_concats; ++n) {
            concat_kernel_arm<OutDataType>(in_concat_axis * _concat_input_size,
                            din + n * in_concat_axis * _concat_input_size,
                            dout + (n * out_concat_axis + offset_concat_axis)
                                       * _concat_input_size);
        }
        offset_concat_axis += in_concat_axis;
    }
    return SaberSuccess;
}

template class SaberConcat<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace anakin

} //namespace anakin

#endif // USE_ARM_PLACE
