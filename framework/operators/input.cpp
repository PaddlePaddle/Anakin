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
    for (int i = 0; i < input_shape.size(); i++) {
        LOG(INFO) << " |-- shape [" << i << "]: " << input_shape[i];
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InputHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
                                              const std::vector<Tensor4dPtr<Ttype>> &ins,
                                              std::vector<Tensor4dPtr<Ttype>> &outs) {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InputHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                               std::vector<Tensor4dPtr<Ttype>> &outs) {
    saber::Shape out_shape;
    for (int i = 0; i < input_shape.size(); i++) {
        out_shape.push_back(input_shape[i]);
    }

    for (auto& tensor_p : outs) {
        tensor_p->set_shape(out_shape);
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

#ifdef USE_X86_PLACE
INSTANCE_INPUT(X86, Precision::FP32);
template class InputHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, X86, Precision::FP32);
#endif

#ifdef USE_AMD
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
#ifdef USE_AMD
    .__alias__<AMD, Precision::FP32>("input")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("input")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("input")
#endif
.Args<PTuple<int>>("input_shape", " shape of graph input.");

} /* namespace ops */

} /* namespace anakin */


