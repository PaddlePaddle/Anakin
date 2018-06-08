#include "framework/operators/input.h"

namespace anakin {

namespace ops {

//#ifdef USE_CUDA
//template<>
//void Input<NV, AK_FLOAT, Precision::FP32>::operator()(OpContext<NV>& ctx,
//        const std::vector<Tensor4dPtr<NV, AK_FLOAT>>& ins,
//        std::vector<Tensor4dPtr<NV, AK_FLOAT>>& outs) {
//}
//#endif
//#ifdef USE_X86_PLACE
//template<>
//void Input<X86, AK_FLOAT, Precision::FP32>::operator()(OpContext<X86>& ctx,
//      const std::vector<Tensor4dPtr<X86, AK_FLOAT>>& ins,
//      std::vector<Tensor4dPtr<X86, AK_FLOAT>>& outs) {
//}
//#endif

/// TODO ... specialization other type of operator
#define INSTANCE_INPUT(Ttype, Dtype, Ptype) \
template<> \
void Input<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
      const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins, \
      std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {}

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
InputHelper<Ttype, Dtype, Ptype>::~InputHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status InputHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Input op parameter.";
    input_shape = GET_PARAMETER(PTuple<int>, input_shape);

    for (int i = 0; i < input_shape.size(); i++) {
        LOG(INFO) << " |-- shape [" << i << "]: " << input_shape[i];
    }

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status InputHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status InputHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
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
INSTANCE_INPUT(NV, AK_FLOAT, Precision::FP32);
INSTANCE_INPUT(NV, AK_FLOAT, Precision::FP16);
INSTANCE_INPUT(NV, AK_FLOAT, Precision::INT8);
template class InputHelper<NV, AK_FLOAT, Precision::FP32>;
template class InputHelper<NV, AK_FLOAT, Precision::FP16>;
template class InputHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, NV, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, NV, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, NV, AK_FLOAT, Precision::INT8);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_INPUT(ARM, AK_FLOAT, Precision::FP32);
template class InputHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
INSTANCE_INPUT(ARM, AK_FLOAT, Precision::FP16);
template class InputHelper<ARM, AK_FLOAT, Precision::FP16>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, ARM, AK_FLOAT, Precision::FP16);
#endif


#ifdef ANAKIN_TYPE_INT8
INSTANCE_INPUT(ARM, AK_FLOAT, Precision::INT8);
template class InputHelper<ARM, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, ARM, AK_FLOAT, Precision::INT8);
#endif

#endif //arm

#ifdef USE_X86_PLACE
INSTANCE_INPUT(X86, AK_FLOAT, Precision::FP32);
INSTANCE_INPUT(X86, AK_FLOAT, Precision::FP16);
INSTANCE_INPUT(X86, AK_FLOAT, Precision::INT8);
template class InputHelper<X86, AK_FLOAT, Precision::FP32>;
template class InputHelper<X86, AK_FLOAT, Precision::FP16>;
template class InputHelper<X86, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, X86, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, X86, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Input, InputHelper, X86, AK_FLOAT, Precision::INT8);
#endif

//! register op
ANAKIN_REGISTER_OP(Input) 
    .Doc("Input operator [ only a input data holder and reshape ] ")
#ifdef USE_CUDA
    .__alias__<NV, AK_FLOAT, Precision::FP32>("input")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, AK_FLOAT, Precision::FP32>("input")
#endif
#ifdef USE_X86_PLACE
    .__alias__<X86, AK_FLOAT, Precision::FP32>("input")
#endif
    .Args<PTuple<int>>("input_shape", " shape of graph input.");

} /* namespace ops */

} /* namespace anakin */


