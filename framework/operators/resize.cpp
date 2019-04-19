#include "framework/operators/resize.h"

namespace anakin {

namespace ops {

#define INSTANCE_RESIZE(Ttype, Ptype) \
template<> \
void Resize<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ResizeHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_resize; \
    impl->_funcs_resize(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ResizeHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Resize op parameter.";

    // get resize param
    auto resize_method = GET_PARAMETER_WITH_DEFAULT(std::string, method,"RESIZE_CUSTOM");
    auto width_scale = GET_PARAMETER_WITH_DEFAULT(float, width_scale, 0.f);
    auto height_scale = GET_PARAMETER_WITH_DEFAULT(float, height_scale, 0.f);
    auto out_width = GET_PARAMETER_WITH_DEFAULT(int, out_width, -1);
    auto out_height = GET_PARAMETER_WITH_DEFAULT(int, out_height, -1);
    if (resize_method == "BILINEAR_ALIGN"){
        ResizeParam<Ttype> resize_param(BILINEAR_ALIGN, height_scale, width_scale, out_width, out_height);
        _param_resize = resize_param;
    } else if (resize_method == "BILINEAR_NO_ALIGN"){
        ResizeParam<Ttype> resize_param(BILINEAR_NO_ALIGN, height_scale, width_scale, out_width, out_height);
         _param_resize = resize_param;
    } else if (resize_method == "RESIZE_CUSTOM"){
        ResizeParam<Ttype> resize_param(RESIZE_CUSTOM, height_scale, width_scale, out_width, out_height);
         _param_resize = resize_param;
    } else if (resize_method == "NEAREST_ALIGN"){
        ResizeParam<Ttype> resize_param(NEAREST_ALIGN, height_scale, width_scale, out_width, out_height);
         _param_resize = resize_param;
    } else {
        LOG(FATAL) << "Resize op doesn't support : " << resize_method << " resize method.";
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ResizeHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                        std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ResizeHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                              std::vector<Tensor4dPtr<Ttype>> &outs) {
    
     auto min_dim = GET_PARAMETER_WITH_DEFAULT(int, min_dim, -1);
    auto max_dim = GET_PARAMETER_WITH_DEFAULT(int, max_dim, -1);
    if (min_dim != -1 && max_dim != -1){
        CHECK_LE(min_dim, max_dim) << "min_dim must less than max_dim";
        int in_h = ins[0] -> height();
        int in_w = ins[0] -> width();
        float in_min = fmin(in_h, in_w);
        float scale = min_dim / in_min;
        int resized_h = int(round(in_h * scale));
        int resized_w = int(round(in_w * scale));
        if (fmax(resized_h, resized_w) > max_dim){
            float in_max = fmax(in_h, in_w);
            scale = max_dim / in_max;
            resized_h = int(round(in_h * scale));
            resized_w = int(round(in_w * scale));            
        }
        ResizeParam<Ttype> resize_param(RESIZE_CUSTOM, scale, scale, resized_w, resized_h);
        _param_resize = resize_param;   
    }

    SABER_CHECK(_funcs_resize.compute_output_shape(ins, outs, _param_resize));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_RESIZE(NV, Precision::FP32);
template <>
Status ResizeHelper<NV, Precision::FP32>::Init(OpContext<NV> &ctx,
                                                        const std::vector<Tensor4dPtr<NV> >& ins,
                                                        std::vector<Tensor4dPtr<NV> >& outs) {
     SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
     return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_RESIZE(X86, Precision::FP32);
template class ResizeHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESIZE(ARM, Precision::FP32);
template class ResizeHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_RESIZE(AMD, Precision::FP32);
template class ResizeHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Resize)
.Doc("Resize operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Resize")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Resize")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("Resize")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Resize")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("method", "resize type to be applied (BILINEAR_ALIGN, BILINEAR_NO_ALIGN, RESIZE_CUSTOM).")
.Args<float>("height_scale", " height scale for resize")
.Args<float>("width_scale", " width scale for resize");

} /* namespace ops */

} /* namespace anakin */
