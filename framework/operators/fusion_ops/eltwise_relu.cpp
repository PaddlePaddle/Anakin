/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "framework/operators/fusion_ops/eltwise_relu.h"

namespace anakin {

namespace ops {

#define INSTANCE_ELTWISERELU(Ttype, Ptype) \
template<> \
void EltwiseRelu<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<EltwiseReluHelper<Ttype, Precision::FP32>*>(this->_helper); \
    auto& param = static_cast<EltwiseReluHelper<Ttype, Precision::FP32>*> \
                  (this->_helper)->_param_eltwise_relu; \
    impl->_funcs_eltwise_relu(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
EltwiseReluHelper<Ttype, Ptype>::~EltwiseReluHelper() {
}

template<typename Ttype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing EltwiseRelu op parameter.";
    auto type = GET_PARAMETER(std::string, type);
    auto alpha = GET_PARAMETER(float, relu_0_alpha);
    auto coeff = GET_PARAMETER(PTuple<float>, coeff);

    ActivationParam<Ttype> activation_param(Active_relu);

    EltwiseType elt_type;

    if (type == "Add") {
        elt_type = Eltwise_sum;
    } else if (type == "Max") {
        elt_type = Eltwise_max;
    } else {
        elt_type = Eltwise_prod;
    }

    //    Shape shape_coeff(1, 1, 1, coeff.size());
    //    Tensor<X86> thcoeff(shape_coeff);
    //    for (int i = 0; i < thcoeff.size(); ++i) {
    //        thcoeff.mutable_data()[i] = coeff[i];
    //    }
    //    Tensor4d<Ttype> * tdcoeff_p = new Tensor4d<Ttype>();
    //    tdcoeff_p->re_alloc(shape_coeff);
    //    tdcoeff_p->copy_from(thcoeff);
    //
    //    saber::EltwiseParam<Ttype>    eltwise_param(elt_type, tdcoeff_p);
    saber::EltwiseParam<Ttype>  eltwise_param(elt_type, coeff.vector(),activation_param);
//    EltwiseActiveParam<Ttype> eltwise_relu_param(eltwise_param, activation_param);
    _param_eltwise_relu =eltwise_param;// eltwise_relu_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_eltwise_relu.init(ins, outs, _param_eltwise_relu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_eltwise_relu.compute_output_shape(ins, outs, _param_eltwise_relu);
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ELTWISERELU(NV, Precision::FP32)
template class EltwiseReluHelper<NV, Precision::FP32>;
template class EltwiseReluHelper<NV, Precision::FP16>;
template class EltwiseReluHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
template class EltwiseReluHelper<ARM, Precision::FP32>;
template class EltwiseReluHelper<ARM, Precision::FP16>;
template class EltwiseReluHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, ARM, Precision::FP32);
#endif

#ifdef BUILD_LITE
INSTANCE_ELTWISERELU(X86, Precision::FP32)
template class EltwiseReluHelper<X86, Precision::FP32>;
template class EltwiseReluHelper<X86, Precision::FP16>;
template class EltwiseReluHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, X86, Precision::FP32);
#endif

// register helper

#ifdef USE_X86_PLACE
INSTANCE_ELTWISERELU(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_ELTWISERELU(AMD, Precision::FP32)
template class EltwiseReluHelper<AMD, Precision::FP32>;
template class EltwiseReluHelper<AMD, Precision::FP16>;
template class EltwiseReluHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(EltwiseRelu)
.Doc("EltwiseRelu operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("eltwise")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("eltwise")
#endif
#ifdef BUILD_LITE
.__alias__<X86, Precision::FP32>("eltwise")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("eltwise")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<float>("relu_0_alpha", " alpha for relu")
.Args<PTuple<float>>("coeff", "coeff of eltwise");

} /* namespace ops */

} /* namespace anakin */


