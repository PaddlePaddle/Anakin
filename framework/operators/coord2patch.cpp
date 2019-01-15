#include "framework/operators/coord2patch.h"

namespace anakin {

namespace ops {

#define INSTANCE_COORD2PATCH(Ttype, Ptype) \
template<> \
void Coord2Patch<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<Coord2PatchHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<Coord2PatchHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_coord2patch; \
    impl->_funcs_coord2patch(ins, outs, param, ctx); \
}
template<typename Ttype, Precision Ptype>
Status Coord2PatchHelper<Ttype, Ptype>::InitParam() {
    auto img_h = GET_PARAMETER(int, img_h);
    auto output_h = GET_PARAMETER(int, output_h);
    auto output_w = GET_PARAMETER(int, output_w);
    saber::Coord2PatchParam<Ttype> param(img_h, output_h, output_w);
    _param_coord2patch = param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Coord2PatchHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_coord2patch.init(ins, outs, _param_coord2patch, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Coord2PatchHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_coord2patch.compute_output_shape(ins, outs, _param_coord2patch));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_COORD2PATCH(NV, Precision::FP32);
template class Coord2PatchHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Coord2Patch, Coord2PatchHelper, NV, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_COORD2PATCH(X86, Precision::FP32);
template class Coord2PatchHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Coord2Patch, Coord2PatchHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_COORD2PATCH(ARM, Precision::FP32);
template class Coord2PatchHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Coord2Patch, Coord2PatchHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Coord2Patch)
.Doc("Coord2Patch operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("coord2patch")
#endif
#ifdef AMD_GPU
//.__alias__<AMD, Precision::FP32>("coord2patch")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("coord2patch")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("coord2patch")
#endif
.num_in(1)
.num_out(1)
.Args<int>("img_h", " img_h for coord2patch ")
.Args<int>("output_h", " output_h for coord2patch ")
.Args<int>("output_w", " output_w for coord2patch ");

} /* namespace ops */

} /* namespace anakin */


