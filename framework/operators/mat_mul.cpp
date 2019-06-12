#include "framework/operators/mat_mul.h"

namespace anakin {

namespace ops {

#define INSTANCE_MAT_MUL(Ttype, Ptype) \
template<> \
void MatMul<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<MatMulHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<MatMulHelper<Ttype, Ptype>*>(this->_helper)->_param_mat_mul; \
    impl->_funcs_mat_mul(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
MatMulHelper<Ttype, Ptype>::~MatMulHelper() {
}

template<typename Ttype, Precision Ptype>
Status MatMulHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing MatMul op parameter.";
    auto transpose_x = GET_PARAMETER(bool, transpose_x);
    auto transpose_y = GET_PARAMETER(bool, transpose_y);
    auto scale = GET_PARAMETER(float, coeff);
    LOG(INFO) <<"mat mul coeff" << scale;
    MatMulParam<Ttype> param_mat_mul(transpose_x, transpose_y, scale);
    _param_mat_mul = param_mat_mul;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MatMulHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_mat_mul.init(ins, outs, _param_mat_mul, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MatMulHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_mat_mul.compute_output_shape(ins, outs, _param_mat_mul));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_MAT_MUL(NV, Precision::FP32);

template<>
Status MatMulHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_mat_mul.init(ins, outs, _param_mat_mul, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_MAT_MUL(X86, Precision::FP32);
INSTANCE_MAT_MUL(X86, Precision::FP16);
INSTANCE_MAT_MUL(X86, Precision::INT8);
template class MatMulHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_MAT_MUL(ARM, Precision::FP32);
template class MatMulHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
//INSTANCE_MAT_MUL(AMD, Precision::FP32);
//template class MatMulHelper<AMD, Precision::FP32>;
//template class MatMulHelper<AMD, Precision::FP16>;
//template class MatMulHelper<AMD, Precision::INT8>;
//ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, AMD, Precision::FP32);
#endif

#ifdef USE_MLU
INSTANCE_MAT_MUL(MLU, Precision::FP32);
INSTANCE_MAT_MUL(MLU, Precision::FP16);
template class MatMulHelper<MLU, Precision::FP32>;
template class MatMulHelper<MLU, Precision::FP16>;
ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, MLU, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(MatMul, MatMulHelper, MLU, Precision::FP16);
#endif

//! register op
ANAKIN_REGISTER_OP(MatMul)
.Doc("MatMul operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("mat_mul")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("mat_mul")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("mat_mul")
#endif
#ifdef AMD_GPU
//.__alias__<AMD, Precision::FP32>("mat_mul")
#endif
#ifdef USE_MLU
.__alias__<MLU, Precision::FP32>("mat_mul")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of MatMul ")
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */

