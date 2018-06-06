#include "framework/operators/scale.h"

namespace anakin {

namespace ops {

template<>
void Scale<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    /*auto* impl = static_cast<Scale<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<Scale<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_permute;
    impl->_funcs_permute(ins, outs, param, ctx);*/
}

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ScaleHelper<Ttype, Dtype, Ptype>::~ScaleHelper() {
    LOG(INFO) << "Decons permute_cpu_float";
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::InitParam() {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    //_funcs_permute.init(ins, outs, _param_permute, SPECIFY, VENDER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    std::vector<Shape4d> shape;

    //_funcs_permute.compute_output_shape(shape, ins, _param_permute);
    //CHECK_EQ(shape.size(), outs.size()) << " size of (out) should be equal to that of vector (shape).";
    for (int i = 0; i <  outs.size(); i++) {
        /// set tensor shape tensor->set_shape(shape[i]);
        outs[i]->set_shape(ins[i]->shape());
    }

    return Status::OK();
}

template class ScaleHelper<NV, AK_FLOAT, Precision::FP32>;
template class ScaleHelper<NV, AK_FLOAT, Precision::FP16>;
template class ScaleHelper<NV, AK_FLOAT, Precision::INT8>;

template class ScaleHelper<ARM, AK_FLOAT, Precision::FP32>;
template class ScaleHelper<ARM, AK_FLOAT, Precision::FP16>;
template class ScaleHelper<ARM, AK_FLOAT, Precision::INT8>;
// register helper
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, NV, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, ARM, AK_FLOAT, Precision::FP32);

//! register op
ANAKIN_REGISTER_OP(Scale)
.Doc("Scale operator")
.__alias__<NV, AK_FLOAT, Precision::FP32>("scale")
.__alias__<ARM, AK_FLOAT, Precision::FP32>("scale")
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


