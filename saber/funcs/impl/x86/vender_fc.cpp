#include "saber/funcs/impl/x86/vender_fc.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

namespace anakin {
namespace saber {

typedef MKL_INT cblas_int;

template class VenderFc<X86, AK_FLOAT>;

template <DataType OpDtype>
SaberStatus VenderFc<X86, OpDtype>
    ::init(const std::vector<Tensor<X86> *>& inputs,
                  std::vector<Tensor<X86> *>& outputs,
                  FcParam<X86> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus VenderFc<X86, OpDtype>
    ::create(const std::vector<Tensor<X86> *>& inputs,
                  std::vector<Tensor<X86> *>& outputs,
                  FcParam<X86> &param, Context<X86> &ctx) {

    //check
    CHECK_EQ(OpDtype, AK_FLOAT) << "vender fc only supports FP32 currently";
    
    this->_ctx = &ctx;
    this->_param = &param;

    MB = inputs[0]->count_valid(0, param.axis);
    OC = outputs[0]->channel();

    // weights
    for (int i = packed_weights.size() - 1; i >= 0; i--) {
       cblas_sgemm_free(packed_weights[i]);
    }
    std::vector<float*> ().swap(packed_weights);

    const float *weights = (const float*)param.weights->data();
    int total_IC = 0;
    for (int i = 0; i < inputs.size(); i++) {
        cblas_int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
        packed_weights.push_back(cblas_sgemm_alloc(CblasAMatrix, OC, MB, IC));
        // LOG(INFO) << "anakin input[" << i << "] alloc passed";
        cblas_sgemm_pack(CblasColMajor,
                         CblasAMatrix,
                         param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                         OC, MB, IC,
                         1.0,
                         weights + total_IC * OC, IC,
                         packed_weights[i]);
        total_IC += IC;
        // LOG(INFO) << "anakin input[" << i << "] pack passed";
    }

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderFc<X86, OpDtype>
    ::dispatch(const std::vector<Tensor<X86> *>& inputs,
                  std::vector<Tensor<X86> *>& outputs,
                  FcParam<X86> &param) {
    
    //check
    CHECK_EQ(OpDtype, AK_FLOAT) << "vender fc only supports FP32 currently";

    float* dst = (float *)outputs[0]->mutable_data();
    const float* bias = NULL;

    if (param.bias) {
        bias = (const float*)param.bias->data();
    }

    for (int i = 0; i < inputs.size(); i++) {
        const float* src = static_cast<const float*>(inputs[i]->data());
        cblas_int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
        if(i == 0) {
            // C := alpha * op(A) * op(B) + beta * C
            cblas_sgemm_compute(CblasColMajor,                                     // Layout
                                CblasPacked,                                       // a
                                CblasNoTrans,                                      // b是否转置
                                OC, MB, IC,                                        // m, n, k
                                packed_weights[i], IC,                             // a, lda
                                src, IC,                                           // b, ldb
                                0.0,                                               // beta
                                dst, OC);                                          // c, ldc
        } else {
            cblas_sgemm_compute(CblasColMajor,                                     // Layout
                                CblasPacked,                                       // a
                                CblasNoTrans,                                      // b是否转置
                                OC, MB, IC,                                        // m, n, k
                                packed_weights[i], IC,                             // a, lda
                                src, IC,                                           // b, ldb
                                1.0,                                               // beta
                                dst, OC);                                          // c, ldc
        }
        //LOG(INFO) << "anakin compute[" << i << "] passed";

        // LOG(INFO) << "inputs[]:dims: " << inputs[0]->dims();
        // LOG(INFO) << "inputs:size: " << inputs.size();
        // LOG(INFO) << "inputs:capacity: " << inputs.capacity();
        // LOG(INFO) << "output:size: " << outputs.size();
        // LOG(INFO) << "OC, MB, IC: " << OC << " "<< MB << " " << IC;
    }

    if (bias) {
        #pragma omp parallel for schedule(static)
        for (cblas_int mb = 0; mb < MB; mb++) {
            cblas_saxpy(OC, 1.0, bias, 1.0, dst + mb * OC, 1);
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(VenderFc, FcParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(VenderFc, FcParam, X86, AK_INT8);
} // namespace saber
} // namespace anakin
