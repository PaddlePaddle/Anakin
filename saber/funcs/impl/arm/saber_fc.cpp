#include "saber/funcs/impl/arm/saber_fc.h"

#ifdef USE_ARM_PLACE

#include "saber/funcs/impl/arm/impl/sgemv_arm.h"

namespace anakin{

namespace saber{

template <typename Dtype>
void fill_bias_fc(Dtype* tensor, const Dtype* bias, const int num, const int channel);
template <>
void fill_bias_fc<float>(float* tensor, const float* bias, const int num, const int channel) {

    int cnt = channel >> 2;
    int remain = channel & 3;

    for (int j = 0; j < num; ++j) {

        const float* ptr_bias = bias;
        float* ptr_out = tensor + j * channel;

        if (cnt > 0) {
            asm(
            ".fill_bias_fc: \n"
            "vld1.32 {d0-d1}, [%[ptr_out]]  @ load data\n"
                    "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
                    "vadd.f32 q2, q0, q1              @ add bias\n"
                    "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
                    "subs   %[cnt], #1                @ loop count -1\n"
                    "bne    .fill_bias_fc             @ jump to main loop\n"
            :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                    [cnt] "+r"(cnt)
            :
            :"q0", "q1", "q2"
            );
        }

        for (; remain > 0; remain--) {
            *(ptr_out++) += *(ptr_bias++);
        }
    }
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberFc<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, FcParam<OpTensor> &param) {

    const InDataType* din = inputs[0]->data();
    OutDataType* dout = outputs[0]->mutable_data();
    const OpDataType* weights = param.weights->data();
    bool flag_bias = param.bias->valid_size() > 0;
    const OpDataType* bias = nullptr;
    if (flag_bias) {
        bias = param.bias->data();
    }

    if (_m > 1 || param.is_transpose_weights) {
        _gemmer(din, _k, weights, (param.is_transpose_weights? _n : _k), dout, _n, 1.f, 0.f, false);
        if (flag_bias) {
            fill_bias_fc(dout, bias, _m, _n);
        }
    } else {
        if (flag_bias) {
            sgemv_bias(false, _n, _k, weights, din, dout, bias);
        } else {
            sgemv(false, _n, _k, weights, din, dout);
        }
    }
    return SaberSuccess;
}

template class SaberFc<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif

