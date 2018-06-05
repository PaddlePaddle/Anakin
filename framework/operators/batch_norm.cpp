#include "framework/operators/batch_norm.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void BatchNorm<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    /*auto* impl = static_cast<BatchNorm<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<BatchNorm<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_permute;
    impl->_funcs_permute(ins, outs, param, ctx);*/
}
#endif
#ifdef USE_ARM_PLACE
template<>
void BatchNorm<ARM, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<ARM>& ctx,
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    /*auto* impl = static_cast<BatchNorm<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<BatchNorm<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_permute;
    impl->_funcs_permute(ins, outs, param, ctx);*/
}
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
BatchNormHelper<Ttype, Dtype, Ptype>::~BatchNormHelper() {
    LOG(INFO) << "Decons permute_cpu_float";
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchNormHelper<Ttype, Dtype, Ptype>::InitParam() {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchNormHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    //_funcs_permute.init(ins, outs, _param_permute, SPECIFY, VENDER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchNormHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    std::vector<Shape4d> shape;

    //_funcs_permute.compute_output_shape(shape, ins, _param_permute);
    //CHECK_EQ(shape.size(), outs.size()) << " size of (out) should be equal to that of vector (shape).";
    for (int i = 0; i <  outs.size(); i++) {
        // set tensor shape tensor->set_shape(shape[i]);
        outs[i]->set_shape(ins[i]->shape());
    }

    return Status::OK();
}
#ifdef USE_CUDA
template class BatchNormHelper<NV, AK_FLOAT, Precision::FP32>;
template class BatchNormHelper<NV, AK_FLOAT, Precision::FP16>;
template class BatchNormHelper<NV, AK_FLOAT, Precision::INT8>;
#endif
<<<<<<< HEAD
=======

>>>>>>> developing
#ifdef USE_ARM_PLACE
template class BatchNormHelper<ARM, AK_FLOAT, Precision::FP32>;
template class BatchNormHelper<ARM, AK_FLOAT, Precision::FP16>;
template class BatchNormHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
<<<<<<< HEAD
=======

>>>>>>> developing
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(BatchNorm)
	.Doc("BatchNorm operator")
#ifdef USE_CUDA
	.__alias__<NV, AK_FLOAT, Precision::FP32>("power")
#endif
#ifdef USE_ARM_PLACE
	.__alias__<ARM, AK_FLOAT, Precision::FP32>("power")
#endif
	.num_in(1)
	.num_out(1)
	.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


