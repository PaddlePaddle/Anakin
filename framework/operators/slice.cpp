#include "framework/operators/slice.h"

namespace anakin {

namespace ops {

#define INSTANCE_SLICE(Ttype, Dtype, Ptype) \
template<> \
void Slice<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<SliceHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SliceHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_slice; \
    impl->_funcs_slice(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SliceHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Slice op parameter.";
    auto slice_dim = GET_PARAMETER(int, slice_dim);
    _slice_point = GET_PARAMETER(PTuple<int>, slice_point);
    _axis = GET_PARAMETER(int, axis);
    LOG(INFO) << " slice_dim " << slice_dim;
    LOG(INFO) << " slice_point size(" << _slice_point.size() << ").";

    for (auto item : _slice_point.vector()) {
        LOG(INFO) << "  |-- " << item;
    }

    LOG(INFO) << " axis " << _axis;

    SliceParam<Tensor4d<Ttype, Dtype>> param_slice(_axis, _slice_point.vector());
    _param_slice = param_slice;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SliceHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_slice.init(ins, outs, _param_slice, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SliceHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    if (_slice_point.size() + 1 != outs.size()) {
        if (_slice_point.size() == 1) {
            for (int i = 0; i < outs.size() - 2; i++) {
                _slice_point.push_back(_slice_point[0] + _slice_point[_slice_point.size() - 1]);
            }

            SliceParam<Tensor4d<Ttype, Dtype>> param_slice(_axis, _slice_point.vector());
            _param_slice = param_slice;
        }
    }

    SABER_CHECK(_funcs_slice.compute_output_shape(ins, outs, _param_slice));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SLICE(NV, AK_FLOAT, Precision::FP32);
template class SliceHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, NV, AK_FLOAT, Precision::FP32);
template class SliceHelper<NV, AK_FLOAT, Precision::FP16>;
template class SliceHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
INSTANCE_SLICE(X86, AK_FLOAT, Precision::FP32);
template class SliceHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SLICE(ARM, AK_FLOAT, Precision::FP32);
template class SliceHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Slice)
.Doc("Slice operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("slice")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("slice")
#endif
.num_in(1)
.num_out(1)
.Args<int>("slice_dim", " slice dim at input ")
.Args<PTuple<int>>("slice_point", " slice point of op")
.Args<int>("axis", " axis of input to slice");

} /* namespace ops */

} /* namespace anakin */


