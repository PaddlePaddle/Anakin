#include "framework/operators/aligned_mat_mul.h"

namespace anakin {

namespace ops {

#define INSTANCE_ALIGNED_MAT_MUL(Ttype, Ptype) \
template<> \
void AlignedMatMul<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<AlignedMatMulHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<AlignedMatMulHelper<Ttype, Ptype>*>(this->_helper)->_param_aligned_mat_mul; \
    impl->_funcs_aligned_mat_mul(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
AlignedMatMulHelper<Ttype, Ptype>::~AlignedMatMulHelper() {
}

template<typename Ttype, Precision Ptype>
Status AlignedMatMulHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing AlignedMatMul op parameter.";
    auto transpose_x = GET_PARAMETER(bool, transpose_x);
    auto transpose_y = GET_PARAMETER(bool, transpose_y);
    auto scale = GET_PARAMETER(float, coeff);
    AlignedMatMulParam<Ttype> param_aligned_mat_mul(transpose_x, transpose_y, scale);
    _param_aligned_mat_mul = param_aligned_mat_mul;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AlignedMatMulHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_aligned_mat_mul.init(ins, outs, _param_aligned_mat_mul, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AlignedMatMulHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_aligned_mat_mul.compute_output_shape(ins, outs, _param_aligned_mat_mul));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ALIGNED_MAT_MUL(NV, Precision::FP32);

template<>
Status AlignedMatMulHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_aligned_mat_mul.init(ins, outs, _param_aligned_mat_mul, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(AlignedMatMul, AlignedMatMulHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ALIGNED_MAT_MUL(X86, Precision::FP32);
INSTANCE_ALIGNED_MAT_MUL(X86, Precision::FP16);
INSTANCE_ALIGNED_MAT_MUL(X86, Precision::INT8);
template class AlignedMatMulHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AlignedMatMul, AlignedMatMulHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ALIGNED_MAT_MUL(ARM, Precision::FP32);
template class AlignedMatMulHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AlignedMatMul, AlignedMatMulHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ALIGNED_MAT_MUL(AMD, Precision::FP32);
template class AlignedMatMulHelper<AMD, Precision::FP32>;
template class AlignedMatMulHelper<AMD, Precision::FP16>;
template class AlignedMatMulHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(AlignedMatMul, AlignedMatMulHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(AlignedMatMul)
.Doc("AlignedMatMul operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("aligned_mat_mul")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("aligned_mat_mul")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("aligned_mat_mul")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("aligned_mat_mul")
#endif
.num_in(2)
.num_out(1)
.Args<bool>("is_transpose_X", "Is X transpose or not")
.Args<bool>("is_transpose_Y", "Is Y transpose or not ")
.Args<float>("scale", "Z = scale * X * Y");

} /* namespace ops */

} /* namespace anakin */

