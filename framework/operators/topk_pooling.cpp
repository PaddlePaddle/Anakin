#include "framework/operators/topk_pooling.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void TopKPooling<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<TopKPoolingHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<TopKPoolingHelper<NV, Precision::FP32>*>(this->_helper)->_param_topk_pooling;
    impl->_funcs_topk_pooling(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void TopKPooling<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<TopKPoolingHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<TopKPoolingHelper<X86, Precision::FP32>*>(this->_helper)->_param_topk_pooling;
    impl->_funcs_topk_pooling(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
TopKPoolingHelper<Ttype, Ptype>::~TopKPoolingHelper() {
}

template<typename Ttype, Precision Ptype>
Status TopKPoolingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing TopKPooling op parameter.";
    auto top_k = GET_PARAMETER(int, top_k);
    auto feat_map_num = GET_PARAMETER(int, feat_map_num);

    TopKPoolingParam<Ttype> param_topk_pooling(top_k, feat_map_num);
    _param_topk_pooling = param_topk_pooling;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status TopKPoolingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_topk_pooling.init(ins, outs, _param_topk_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status TopKPoolingHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_topk_pooling.compute_output_shape(ins, outs, _param_topk_pooling));
    return Status::OK();
}

#ifdef USE_CUDA
template class TopKPoolingHelper<NV, Precision::FP32>;
template class TopKPoolingHelper<NV, Precision::FP16>;
template class TopKPoolingHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class TopKPoolingHelper<ARM, Precision::FP32>;
template class TopKPoolingHelper<ARM, Precision::FP16>;
template class TopKPoolingHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class TopKPoolingHelper<X86, Precision::FP32>;
template class TopKPoolingHelper<X86, Precision::FP16>;
template class TopKPoolingHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(TopKPooling, TopKPoolingHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(TopKPooling, TopKPoolingHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(TopKPooling, TopKPoolingHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(TopKPooling)
.Doc("TopKPooling operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("topk_pooling")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("topk_pooling")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("topk_pooling")
#endif
.num_in(1)
.num_out(1)
.Args<int>("top_k", "get top k max data of each feature map")
.Args<int>("feat_map_num", "feature map num");

} /* namespace ops */

} /* namespace anakin */


