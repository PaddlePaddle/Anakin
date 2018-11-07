#include "framework/operators/pooling.h"

namespace anakin {

namespace ops {

#define INSTANCE_POOLING(Ttype, Ptype) \
template<> \
void Pooling<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PoolingHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PoolingHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_pooling; \
    impl->_funcs_pooling(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status PoolingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Pooling op parameter.";
    auto cmp_out_shape_floor_as_conv = GET_PARAMETER(bool, cmp_out_shape_floor_as_conv);
    auto global_pooling = GET_PARAMETER(bool, global_pooling);
    auto pool_padding = GET_PARAMETER(PTuple<int>, padding);
    auto pool_strides = GET_PARAMETER(PTuple<int>, strides);
    auto pool_size = GET_PARAMETER(PTuple<int>, pool_size);
    auto pool_method = GET_PARAMETER(std::string, method);

    if (pool_method == "MAX") {
        PoolingParam<Ttype> pooling_param(pool_size[0], pool_size[1],
                                          pool_padding[0], pool_padding[1],
                                          pool_strides[0], pool_strides[1],
                                          Pooling_max, global_pooling, cmp_out_shape_floor_as_conv);
        _param_pooling = pooling_param;
    } else if (pool_method == "AVG") {
        PoolingParam<Ttype> pooling_param(pool_size[0], pool_size[1],
                                          pool_padding[0], pool_padding[1],
                                          pool_strides[0], pool_strides[1],
                                          Pooling_average_include_padding, global_pooling, cmp_out_shape_floor_as_conv);
        _param_pooling = pooling_param;
    } else {
        LOG(FATAL) << " Pooling op doesn't support : " << pool_method << " pooling.";
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PoolingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_pooling.init(ins, outs, _param_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PoolingHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_pooling.compute_output_shape(ins, outs, _param_pooling));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_POOLING(NV, Precision::FP32);
template <>
Status PoolingHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, \
        const std::vector<Tensor4dPtr<NV> >& ins, \
        std::vector<Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_pooling.init(ins, outs, _param_pooling, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_POOLING(ARM, Precision::FP32);
template class PoolingHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, ARM, Precision::FP32);
#endif  //arm

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_POOLING(X86, Precision::FP32);
template class PoolingHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_POOLING(AMD, Precision::FP32);
template class PoolingHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Pooling)
.Doc("Pooling operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("pooling")
.__alias__<NV, Precision::FP32>("pool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("pooling")
.__alias__<ARM, Precision::FP32>("pool")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("pooling")
.__alias__<X86, Precision::FP32>("pool")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("pooling")
.__alias__<AMD, Precision::FP32>("pool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("method", "Pooling type to be applied (MAX, SUM, AVG).")
.Args<bool>("cmp_out_shape_floor_as_conv cmp_out_shape_floor_as_conv of pooling for adu novel approach")
.Args<bool>("global_pooling", "whether execute global pooling on input")
.Args<PTuple<int>>("pool_size", " kernel size for pooling (x, y) or (x, y, z).")
                .Args<PTuple<int>>("strides",  "stride for pooling (x, y)  or  (x, y, z).")
                .Args<PTuple<int>>("padding", "pad for pooling: (x, y) or (x, y, z).");

} /* namespace ops */

} /* namespace anakin */


