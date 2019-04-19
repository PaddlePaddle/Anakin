#include "framework/operators/pad2d.h"

namespace anakin {

namespace ops {

#define INSTANCE_PAD2D(Ttype, Ptype) \
template<> \
void Pad2D<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
	auto* impl = static_cast<Pad2DHelper<Ttype, Ptype>*>(this->_helper); \
	auto& param = static_cast<Pad2DHelper<Ttype, Ptype>*> \
	              (this->_helper)->_param_pad2d; \
	impl->_funcs_pad2d(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status Pad2DHelper<Ttype, Ptype>::InitParam() {
	DLOG(WARNING) << "Parsing Pad2D op parameter.";
	auto mode = GET_PARAMETER(std::string, mode);
	auto pad_value = GET_PARAMETER_WITH_DEFAULT(float, value, 0.f);
	auto pad_h = GET_PARAMETER(PTuple<int>, pad_h);
	auto pad_w = GET_PARAMETER(PTuple<int>, pad_w);

	PadMode pad_mode;
	if (mode == "constant"){
		pad_mode = PAD_CONSTANT;
	} else if (mode == "edge"){
		pad_mode = PAD_EDGE;
	} else if (mode == "reflect"){
		pad_mode = PAD_REFLECT;
	} else {
		pad_mode = PAD_CONSTANT;
	}
	saber::Pad2DParam<Ttype> pad2d_param(pad_h.vector(), pad_w.vector(), pad_value, pad_mode);
	_param_pad2d = pad2d_param;
	return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Pad2DHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                     const std::vector<Tensor4dPtr<Ttype> >& ins,
                                     std::vector<Tensor4dPtr<Ttype> >& outs) {
	SABER_CHECK(_funcs_pad2d.init(ins, outs, _param_pad2d, SPECIFY, SABER_IMPL, ctx));
	return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Pad2DHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&ins,
                                           std::vector<Tensor4dPtr<Ttype> >& outs) {
	SABER_CHECK(_funcs_pad2d.compute_output_shape(ins, outs, _param_pad2d));
	return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PAD2D(NV, Precision::FP32);
template class Pad2DHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad2D, Pad2DHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_PAD2D(AMD, Precision::FP32);
template class Pad2DHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad2D, Pad2DHelper, AMD, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_PAD2D(X86, Precision::FP32);
template class Pad2DHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad2D, Pad2DHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PAD2D(ARM, Precision::FP32);
template class Pad2DHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad2D, Pad2DHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Pad2D)
.Doc("Pad2D operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Pad2D")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Pad2D")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("Pad2D")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Pad2D")
#endif
.num_in(1)
.num_out(1)
.Args<int>("mode", "pad mode")
.Args<float>("pad_value", "pad value")
.Args<PTuple<int>>("pad_h", "pad left and right value")
.Args<PTuple<int>>("pad_w", "pad top and bottom value");

} /* namespace ops */

} /* namespace anakin */
