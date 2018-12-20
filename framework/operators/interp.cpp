#include "framework/operators/interp.h"
#include "saber/funcs/resize.h"

namespace anakin {

namespace ops {

#define INSTANCE_INTERP(Ttype, Ptype) \
template<> \
void Interp<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<InterpHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_resize; \
    impl->_funcs_resize(ins, outs, param, ctx); \
    auto shape_n = outs[0]->shape(); \
    for (int i = 1; i < outs.size(); ++i){ \
        outs[i]->re_alloc(shape_n); \
        outs[i]->copy_from(*outs[0]);\
    } \
}

template<typename Ttype, Precision Ptype>
Status InterpHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Resize op parameter.";
    // get interp param
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InterpHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                        std::vector<Tensor4dPtr<Ttype>> &outs) {
    // get interp param
    SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status InterpHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                              std::vector<Tensor4dPtr<Ttype>> &outs) {
    float width_scale = 1.0f;
    float height_scale = 1.0f;
    float width_in = ins[0]->width();
    float height_in = ins[0]->height();
    int ho = 1;
    int wo= 1;
    auto shrink_factor = GET_PARAMETER(int, shrink_factor);
    ho = (height_in-1)/shrink_factor+1;
    wo = (width_in-1)/shrink_factor+1;

    auto zoom_factor = GET_PARAMETER(int, zoom_factor);
    ho = ho+(ho-1)*(zoom_factor-1);
    wo = wo+(wo-1)*(zoom_factor-1);
    float height_out = GET_PARAMETER(int, height);
    float width_out = GET_PARAMETER(int, width);
    if (height_out !=0 && width_out!=0){
        width_scale = width_out/width_in;
        height_scale = height_out/height_in;
    } else {
        width_scale = wo/width_in;
        height_scale = ho/height_in;

    }
    LOG(INFO)<<"height_out:"<<height_out<<"height_in"<<height_in;
    LOG(INFO)<<"width_out:"<<width_out<<"width_in"<<width_in;
    SET_PARAMETER(width_scale, width_scale, float);
    SET_PARAMETER(height_scale, height_scale, float);
    LOG(INFO)<<height_scale <<"<->"<<width_scale;
    //transfer
    ResizeParam<Ttype> resize_param(RESIZE_CUSTOM, width_scale, height_scale);
    _param_resize = resize_param;

    SABER_CHECK(_funcs_resize.compute_output_shape(ins, outs, _param_resize));
    auto shape_n = outs[0]->shape();
    auto shape_in = ins[0]->shape();
        LOG(INFO)<<"shape0"<<shape_n[0]<<" "<<shape_n[1]<<" "<<shape_n[2]<<" "<<shape_n[3];
        LOG(INFO)<<"shapeIN"<<shape_in[0]<<" "<<shape_in[1]<<" "<<shape_in[2]<<" "<<shape_in[3];
    for (int i = 1; i < outs.size(); ++i){
        outs[i]->set_shape(outs[0]->shape());
        LOG(INFO)<<"shape"<<i<<":"<<shape_n[0]<<" "<<shape_n[1]<<" "<<shape_n[2]<<" "<<shape_n[3];
    }
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_INTERP(NV, Precision::FP32);
template <>
Status InterpHelper<NV, Precision::FP32>::Init(OpContext<NV> &ctx,
                                                        const std::vector<Tensor4dPtr<NV> >& ins,
                                                        std::vector<Tensor4dPtr<NV> >& outs) {
     SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
     return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Interp, InterpHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined(BUILD_LITE)
INSTANCE_INTERP(X86, Precision::FP32);
template class InterpHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Interp, InterpHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_INTERP(ARM, Precision::FP32);
template class InterpHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Interp, InterpHelper, ARM, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(Interp)
.Doc("Interp operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Interp")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Interp")
#endif
#if defined USE_X86_PLACE || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("Interp")
#endif
.num_in(1)
.num_out(1)
.Args<float>("height_scale", " height scale for resize")
.Args<float>("width_scale", " width scale for resize");

} /* namespace ops */

} /* namespace anakin */
