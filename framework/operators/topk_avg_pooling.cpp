#include "framework/operators/topk_avg_pooling.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void TopKAvgPooling<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<TopKAvgPoolingHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<TopKAvgPoolingHelper<NV, Precision::FP32>*>(this->_helper)->_param_topk_avg_pooling;
    impl->_funcs_topk_avg_pooling(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void TopKAvgPooling<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<TopKAvgPoolingHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<TopKAvgPoolingHelper<X86, Precision::FP32>*>(this->_helper)->_param_topk_avg_pooling;
    impl->_funcs_topk_avg_pooling(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
TopKAvgPoolingHelper<Ttype, Ptype>::~TopKAvgPoolingHelper() {
}

template<typename Ttype, Precision Ptype>
Status TopKAvgPoolingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing TopKAvgPooling op parameter.";
    auto top_ks = GET_PARAMETER(PTuple<int>, top_ks);
    auto feat_map_num = GET_PARAMETER(int, feat_map_num);
    auto is_pooling_by_row = GET_PARAMETER(bool, is_pooling_by_row);

    TopKAvgPoolingParam<Ttype> param_topk_avg_pooling(top_ks.vector(),
            feat_map_num, is_pooling_by_row);
    _param_topk_avg_pooling = param_topk_avg_pooling;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status TopKAvgPoolingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_topk_avg_pooling.init(ins, outs, _param_topk_avg_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status TopKAvgPoolingHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_topk_avg_pooling.compute_output_shape(ins, outs, _param_topk_avg_pooling));
    return Status::OK();
}

#ifdef USE_CUDA
template class TopKAvgPoolingHelper<NV, Precision::FP32>;
template class TopKAvgPoolingHelper<NV, Precision::FP16>;
template class TopKAvgPoolingHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class TopKAvgPoolingHelper<ARM, Precision::FP32>;
template class TopKAvgPoolingHelper<ARM, Precision::FP16>;
template class TopKAvgPoolingHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class TopKAvgPoolingHelper<X86, Precision::FP32>;
template class TopKAvgPoolingHelper<X86, Precision::FP16>;
template class TopKAvgPoolingHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(TopKAvgPooling, TopKAvgPoolingHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(TopKAvgPooling, TopKAvgPoolingHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(TopKAvgPooling, TopKAvgPoolingHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(TopKAvgPooling)
.Doc("TopKAvgPooling operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("topk_avg_pooling")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("topk_avg_pooling")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("topk_avg_pooling")
#endif
.num_in(1)
.num_out(1)
.Args<int>("feat_map_num", "feat map nums")
.Args<bool>("is_pooling_by_row", "pooling by row if true else by col");

} /* namespace ops */

} /* namespace anakin */


