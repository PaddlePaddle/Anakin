#include "framework/operators/detection_output.h"

namespace anakin {

namespace ops {

#define INSTANCE_DETECTIONOUTPUT(Ttype, Ptype) \
template<> \
void DetectionOutput<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<DetectionOutputHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DetectionOutputHelper<Ttype, Ptype>*>(this->_helper)->_param_detection_output; \
    impl->_funcs_detection_output(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status DetectionOutputHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Detectionoutput op parameter.";
    auto flag_share_location = GET_PARAMETER(bool, share_location);
    auto flag_var_in_target  = GET_PARAMETER(bool, variance_encode_in_target);
    auto classes_num         = GET_PARAMETER(int, class_num);
    auto background_id_      = GET_PARAMETER(int, background_id);
    auto keep_top_k_         = GET_PARAMETER(int, keep_top_k);
    auto code_type_          = GET_PARAMETER(std::string, code_type);
    auto conf_thresh_        = GET_PARAMETER(float, conf_thresh);
    auto nms_top_k_          = GET_PARAMETER(int, nms_top_k);
    auto nms_thresh_         = GET_PARAMETER(float, nms_thresh);
    auto nms_eta_            = GET_PARAMETER(float, nms_eta);

    CodeType code_type = CENTER_SIZE;

    if (code_type_ == "CORNER") {
        code_type = CORNER;
    } else if (code_type_ == "CENTER_SIZE") {
        code_type = CENTER_SIZE;
    } else if (code_type_ == "CORNER_SIZE") {
        code_type = CORNER_SIZE;
    } else {
                LOG(FATAL) << "unsupport type: " << code_type_;
    }

    DetectionOutputParam<Ttype> param_det(classes_num, background_id_, \
            keep_top_k_, nms_top_k_, nms_thresh_, conf_thresh_, \
            flag_share_location, flag_var_in_target, code_type, nms_eta_);
    _param_detection_output = param_det;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DetectionOutputHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                                   std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_detection_output.init(ins, outs, _param_detection_output, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DetectionOutputHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                         std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_detection_output.compute_output_shape(ins, outs, _param_detection_output));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_DETECTIONOUTPUT(NV, Precision::FP32);
template class DetectionOutputHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DetectionOutput, DetectionOutputHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_DETECTIONOUTPUT(X86, Precision::FP32);
template class DetectionOutputHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DetectionOutput, DetectionOutputHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DETECTIONOUTPUT(ARM, Precision::FP32);
template class DetectionOutputHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DetectionOutput, DetectionOutputHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(DetectionOutput)
.Doc("DetectionOutput operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("detectionoutput")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("detectionoutput")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("detectionoutput")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("share_location", " flag whether all classes share location ")
.Args<bool>("variance_encode_in_target", " flag whether variance is encode in location ")
.Args<int>("class_num", " number of classes to detect ")
.Args<int>("background_id", " background id")
.Args<int>("keep_top_k", " number to keep in detectoin ")
.Args<std::string>("code_type", " bbox code type ")
.Args<float>("conf_thresh", " confidece threshold in detection ")
.Args<int>("nms_top_k", " number to keep in nms ")
.Args<float>("nms_thresh", " overlap threshold for nms ")
.Args<float>("nms_eta", " eta for nms ");

} /* namespace ops */

} /* namespace anakin */


