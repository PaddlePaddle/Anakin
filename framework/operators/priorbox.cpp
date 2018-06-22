#include "framework/operators/priorbox.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void PriorBox<NV, AK_FLOAT, Precision::FP32>::operator()(OpContext<NV>& ctx, \
        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, \
        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<PriorBoxHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<PriorBoxHelper<NV, AK_FLOAT, \
                  Precision::FP32>*>(this->_helper)->_param_priorbox;
    impl->_funcs_priorbox(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
PriorBoxHelper<Ttype, Dtype, Ptype>::~PriorBoxHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PriorBoxHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PriorBox op parameter.";
    auto min_size_ = GET_PARAMETER(PTuple<float>, min_size);
    auto max_size_ = GET_PARAMETER(PTuple<float>, max_size);
    auto as_ratio  = GET_PARAMETER(PTuple<float>, aspect_ratio);
    auto flip_flag = GET_PARAMETER(bool, is_flip);
    auto clip_flag = GET_PARAMETER(bool, is_clip);
    auto var       = GET_PARAMETER(PTuple<float>, variance);
    auto image_h   = GET_PARAMETER(int, img_h);
    auto image_w   = GET_PARAMETER(int, img_w);
    auto step_h_   = GET_PARAMETER(float, step_h);
    auto step_w_   = GET_PARAMETER(float, step_w);
    auto offset_   = GET_PARAMETER(float, offset);
    auto order     = GET_PARAMETER(PTuple<std::string>, order);
    std::vector<PriorType> order_;

    for (int i = 0; i < order.size(); i++) {
        if (order[i] == "MIN") {
            order_.push_back(PRIOR_MIN);
        } else if (order[i] == "MAX") {
            order_.push_back(PRIOR_MAX);
        } else if (order[i] == "COM") {
            order_.push_back(PRIOR_COM);
        }
    }

    saber::PriorBoxParam<Tensor4d<Ttype, Dtype>> param_priorbox(min_size_.vector(), max_size_.vector(), \
                                       as_ratio.vector(), var.vector(), flip_flag, clip_flag, \
                                       image_w, image_h, step_w_, step_h_, offset_, order_);
    _param_priorbox = param_priorbox;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PriorBoxHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_priorbox.init(ins, outs, _param_priorbox, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PriorBoxHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_priorbox.compute_output_shape(ins, outs, _param_priorbox));
    return Status::OK();
}

#ifdef USE_CUDA
template class PriorBoxHelper<NV, AK_FLOAT, Precision::FP32>;
template class PriorBoxHelper<NV, AK_FLOAT, Precision::FP16>;
template class PriorBoxHelper<NV, AK_FLOAT, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class PriorBoxHelper<ARM, AK_FLOAT, Precision::FP32>;
template class PriorBoxHelper<ARM, AK_FLOAT, Precision::FP16>;
template class PriorBoxHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(PriorBox, PriorBoxHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(PriorBox, PriorBoxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(PriorBox)
.Doc("PriorBox operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("priorbox")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("priorbox")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<float>>("min_size", " min_size of bbox ")
                  .Args<PTuple<float>>("max_size", " max_size of bbox ")
                  .Args<PTuple<float>>("aspect_ratio", " aspect ratio of bbox ")
                  .Args<bool>("is_flip", "flip flag of bbox")
                  .Args<bool>("is_clip", "clip flag of bbox")
                  .Args<PTuple<float>>("variance", " variance of bbox ")
                  .Args<int>("img_h", "input image height")
                  .Args<int>("img_w", "input image width")
                  .Args<float>("step_h", "height step of bbox")
                  .Args<float>("step_w", "width step of bbox")
                  .Args<float>("offset", "center offset of bbox");

} /* namespace ops */

} /* namespace anakin */


