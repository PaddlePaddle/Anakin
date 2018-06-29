#include "framework/operators/permute.h"

namespace anakin {

namespace ops {

#define INSTANCE_PERMUTE(Ttype, Dtype, Ptype) \
template<> \
void Permute<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<PermuteHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PermuteHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_permute; \
    impl->_funcs_permute(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermuteHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "!!!!!!!! Parsing Permute op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);

    for (int i = 0; i < dims.size(); i++) {
        LOG(INFO) << " |-- dims [" << i << "]: " << dims[i];
    }

    saber::PermuteParam<Tensor4d<Ttype, Dtype>> permute_param(dims.vector());
    _param_permute = permute_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermuteHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_permute.init(ins, outs, _param_permute, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermuteHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_permute.compute_output_shape(ins, outs, _param_permute));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PERMUTE(NV, AK_FLOAT, Precision::FP32);
template class PermuteHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_PERMUTE(X86, AK_FLOAT, Precision::FP32);
template class PermuteHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PERMUTE(ARM, AK_FLOAT, Precision::FP32);
template class PermuteHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Permute)
.Doc("Permute operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("permute")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("permute")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("permute")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


