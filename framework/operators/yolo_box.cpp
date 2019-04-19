#include "framework/operators/yolo_box.h"

namespace anakin {

namespace ops {

#define INSTANCE_YOLO_BOX(Ttype, Ptype) \
template<> \
void YoloBox<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<YoloBoxHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
    static_cast<YoloBoxHelper<Ttype, Ptype>*>(this->_helper)->_param_yolo_box; \
    impl->_funcs_yolo_box(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
YoloBoxHelper<Ttype, Ptype>::~YoloBoxHelper() {
}

template<typename Ttype, Precision Ptype>
Status YoloBoxHelper<Ttype, Ptype>::InitParam() {
            DLOG(WARNING) << "Parsing YoloBox op parameter.";
    auto anchors = GET_PARAMETER(PTuple<int>, anchors);
    auto class_num = GET_PARAMETER(int, class_num);
    auto conf_thresh = GET_PARAMETER(float, conf_thresh);
    auto downsample_ratio = GET_PARAMETER(int, downsample_ratio);
    YoloBoxParam<Ttype> param_yolo_box(anchors.vector(), class_num, conf_thresh, downsample_ratio);
    _param_yolo_box = param_yolo_box;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status YoloBoxHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_yolo_box.init(ins, outs, _param_yolo_box, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status YoloBoxHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_yolo_box.compute_output_shape(ins, outs, _param_yolo_box));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_YOLO_BOX(NV, Precision::FP32);
template class YoloBoxHelper<NV, Precision::FP32>;
template class YoloBoxHelper<NV, Precision::FP16>;
template class YoloBoxHelper<NV, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
INSTANCE_YOLO_BOX(X86, Precision::FP32);
template class YoloBoxHelper<X86, Precision::FP32>;
template class YoloBoxHelper<X86, Precision::FP16>;
template class YoloBoxHelper<X86, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
INSTANCE_YOLO_BOX(ARM, Precision::FP32);
template class YoloBoxHelper<ARM, Precision::FP32>;
template class YoloBoxHelper<ARM, Precision::FP16>;
template class YoloBoxHelper<ARM, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(YoloBox, YoloBoxHelper, NV, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(YoloBox, YoloBoxHelper, X86, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(YoloBox, YoloBoxHelper, ARM, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(YoloBox)
.Doc("YoloBox operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("yolo_box")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("yolo_box")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("yolo_box")
#endif
.num_in(2)
.num_out(2)
.Args<PTuple<int>>("anchors", "anchor of yolo_box_param")
.Args<int>("class_num", "get class_num")
.Args<float>("conf_thresh", "conf_thresh map num")
.Args<int>("downsample_ratio", "get downsample_ratio");

} /* namespace ops */

} /* namespace anakin */


