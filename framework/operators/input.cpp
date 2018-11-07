#include "framework/operators/input.h"

namespace anakin {

namespace ops {

#define INSTANCE_INPUT(Ttype, Ptype) \
template<> \
void Input<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
      const std::vector<Tensor4dPtr<Ttype>>& ins, \
      std::vector<Tensor4dPtr<Ttype>>& outs) {}


template<typename Ttype, Precision Ptype>
Status InputHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Input op parameter.";
    input_shape = GET_PARAMETER(PTuple<int>, input_shape);

    if (CHECK_PARAMETER(max_len)) {
        max_len = GET_PARAMETER(int , max_len);
    }

    if (CHECK_PARAMETER(max_batch)) {
        max_batch = GET_PARAMETER(int , max_batch);
    }

    for (int i = 0; i < input_shape.size(); i++) {
        LOG(INFO) << " |-- shape [" << i << "]: " << input_shape[i];
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InputHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                       const std::vector<Tensor4dPtr<Ttype>>& ins,
                                       std::vector<Tensor4dPtr<Ttype>>& outs) {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InputHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    saber::Shape out_shape;

    for (int i = 0; i < input_shape.size(); i++) {
        out_shape.push_back(input_shape[i]);
    }

    for (auto & tensor_p : outs) {
        tensor_p->set_shape(out_shape);
    }

    if (max_len != 0 && max_batch != 0) {
        std::vector<std::vector<int>> seq_offset(1, std::vector<int>(max_batch + 1, 0));
        int i;

        for (i = 0; i < max_batch; i++) {
            seq_offset[0][i] = i * max_len;
        }

        seq_offset[0][i] = i * max_len;

        for (auto & tensor_p : outs) {
            tensor_p->set_seq_offset(seq_offset);
        }
    }

    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_INPUT(NV, Precision::FP32);
template class InputHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_INPUT(ARM, Precision::FP32);
template class InputHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, ARM, Precision::FP32);
#endif //arm

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_INPUT(X86, Precision::FP32);
template class InputHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_INPUT(AMD, Precision::FP32);
template class InputHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Input)
.Doc("Input operator [ only a input data holder and reshape ] ")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("input")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("input")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("input")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("input")
#endif
.Args<PTuple<int>>("input_shape", " shape of graph input.");

} /* namespace ops */

} /* namespace anakin */


