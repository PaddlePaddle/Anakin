#include "framework/operators/crop.h"

namespace anakin {

namespace ops {

#define INSTANCE_CROP(Ttype, Ptype) \
template<> \
void Crop<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<CropHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<CropHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_crop; \
    impl->_funcs_crop(ins, outs, param, ctx); \
}
/// set helper
template<typename Ttype, Precision Ptype>
CropHelper<Ttype, Ptype>::~CropHelper() {
}

template<typename Ttype, Precision Ptype>
Status CropHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Crop op parameter.";

	using pblock_type = PBlock<Ttype>;
    auto axis = GET_PARAMETER(int, axis);
    auto offset_in = GET_PARAMETER(PTuple<int>, cropping);
    std::vector<int> shape;
    shape.push_back(axis);
    for(int i = 0; i < offset_in.size(); i++){
        shape.push_back(offset_in[i]);
    }
    saber::CropParam<Ttype> crop_param(axis, offset_in.vector(), shape);
    _param_crop = crop_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CropHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_crop.init(ins, outs, _param_crop, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CropHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_crop.compute_output_shape(ins, outs, _param_crop));
    return Status::OK();
}

#ifdef USE_CUDA
template class CropHelper<NV, Precision::FP32>;
template class CropHelper<NV, Precision::FP16>;
template class CropHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CropHelper<ARM, Precision::FP32>;
template class CropHelper<ARM, Precision::FP16>;
template class CropHelper<ARM, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined(BUILD_LITE)
template class CropHelper<X86, Precision::FP32>;
template class CropHelper<X86, Precision::FP16>;
template class CropHelper<X86, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
INSTANCE_CROP(NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Crop, CropHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CROP(ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Crop, CropHelper, ARM, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined(BUILD_LITE)
INSTANCE_CROP(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Crop, CropHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Crop)
.Doc("Crop operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Crop")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Crop")
#endif
#if defined (USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("Crop")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", "axis of crop")
.Args<PTuple<int>>("offset", "offset_in crop");

} /* namespace ops */

} /* namespace anakin */


