#include "framework/operators/priorbox.h"

namespace anakin {

namespace ops {

#define INSTANCE_PRIORBOX(Ttype, Dtype, Ptype) \
template<> \
void PriorBox<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<PriorBoxHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PriorBoxHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_priorbox; \
    impl->_funcs_priorbox(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PriorBoxHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PriorBox op parameter.";
    auto min_size_ = GET_PARAMETER(PTuple<float>, min_size);
    //add new parameter
    PTuple<float> fixed_size_;
    if (FIND_PARAMETER(fixed_size)) {
        fixed_size_ = GET_PARAMETER(PTuple<float>, fixed_size);
    }
   // LOG(ERROR) << "fixed_size size " << fixed_size_.size();
    PTuple<float> fixed_ratio_;
    if (FIND_PARAMETER(fixed_ratio)) {
        fixed_ratio_ = GET_PARAMETER(PTuple<float>, fixed_ratio);;
    }
    //LOG(ERROR) << "fixed_ratio size " << fixed_ratio_.size();
    PTuple<float> density_;
    if (FIND_PARAMETER(density)) {
         density_ = GET_PARAMETER(PTuple<float>, density);
    }
    //LOG(ERROR) << "density_ size " << density_.size();
    //end
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

    if(min_size_.size() <= 0){//min
      saber::PriorBoxParam<Tensor4d<Ttype, Dtype>> param_priorbox( var.vector(), flip_flag, clip_flag, \
                                       image_w, image_h, step_w_, step_h_, offset_, order_, \
                                       std::vector<float>(), std::vector<float>(), std::vector<float>(), \
                                       fixed_size_.vector(), fixed_ratio_.vector(), density_.vector());
       _param_priorbox = param_priorbox;
    }else{
      saber::PriorBoxParam<Tensor4d<Ttype, Dtype>> param_priorbox(var.vector(), flip_flag, clip_flag, \
                                       image_w, image_h, step_w_, step_h_, offset_, order_,  \
                                       min_size_.vector(), max_size_.vector(), as_ratio.vector(), \
                                       std::vector<float>(), std::vector<float>(), std::vector<float>());
       _param_priorbox = param_priorbox;
    }
    
  //  saber::PriorBoxParam<Tensor4d<Ttype, Dtype>> param_priorbox(min_size_.vector(), max_size_.vector(), \
                                       as_ratio.vector(), var.vector(), flip_flag, clip_flag, \
                                       image_w, image_h, step_w_, step_h_, offset_, order_, \
                                       fixed_size_.vector(), fixed_ratio_.vector(), density_.vector());
   // _param_priorbox = param_priorbox;
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
INSTANCE_PRIORBOX(NV, AK_FLOAT, Precision::FP32);
template class PriorBoxHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PriorBox, PriorBoxHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PRIORBOX(ARM, AK_FLOAT, Precision::FP32);
template class PriorBoxHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PriorBox, PriorBoxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_PRIORBOX(X86, AK_FLOAT, Precision::FP32);
template class PriorBoxHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PriorBox, PriorBoxHelper, X86, AK_FLOAT, Precision::FP32);
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
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("priorbox")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<float>>("min_size", " min_size of bbox ")
                  .Args<PTuple<float>>("fixed_size", " fixed_size of bbox ")
                  .Args<PTuple<float>>("fixed_ratio", " fixed_ratio of bbox ")
                  .Args<PTuple<float>>("density_", " density_ of bbox ")
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


