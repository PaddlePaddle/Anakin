#include "saber/funcs/impl/x86/vender_fc.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

namespace anakin {
namespace saber {

typedef MKL_INT cblas_int;

template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<X86> &ctx) {
    this->_ctx = ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::create(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<X86> &ctx) {
    if (inDtype != AK_FLOAT) {
        LOG(ERROR) << "vender fc only supports FP32 currently";
        return SaberUnImplError;
    }

    this->_ctx = ctx;
    this->_param = &param;

    // bias
    if (this->bias_sum) {
        free(this->bias_sum);
        this->bias_sum = nullptr;
    }

    MB = inputs[0]->count_valid(0, param.axis);
    OC = outputs[0]->channel();

    if (param.bias && inputs.size() > 1) {
        this->bias_sum = (float*)zmalloc(OC * sizeof(float), 4096);
        if (!this->bias_sum) {
            LOG(ERROR) << "out of memory in vender fc";
            return SaberOutOfMem;
        }
    }

    // weights
    for (int i = packed_weights.size() - 1; i >= 0; i--) {
       DataType_op *pw = packed_weights[i];
       cblas_sgemm_free(pw);
       pw = nullptr;
       packed_weights.pop_back();
    }
    std::vector<DataType_op*> ().swap(packed_weights);

    const DataType_op* weights = param.weights->data();
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

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param) {
    if (inDtype == AK_FLOAT) {
        //LOG(INFO) << "anakin dispatch come in";
        float* dst = static_cast<float*>(outputs[0]->mutable_data());
        const float* bias = NULL;

        if (param.bias) {
            bias = static_cast<const float*>(param.bias->data());
        }

        if (bias_sum) {
            memset(bias_sum, 0, OC * sizeof(float));
        }

        for (int i = 0; i < inputs.size(); i++) {
            const float* src = static_cast<const float*>(inputs[i]->data());
            cblas_int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
            if(i == 0) {
                // C := alpha * op(A) * op(B) + beta * C
                cblas_sgemm_compute(CblasColMajor,                                     // 二维数组Layout
                                    CblasPacked,                                       // a
                                    CblasNoTrans,                                      // b是否转置
                                    OC, MB, IC,                                        // m, n, k
                                    packed_weights[i], IC,                             // a, lda
                                    src, IC,                                           // b, ldb
                                    0.0,                                               // beta
                                    dst, OC);                                          // c, ldc
            } else {
                cblas_sgemm_compute(CblasColMajor,                                     // 二维数组Layout
                                    CblasPacked,                                       // a
                                    CblasNoTrans,                                      // b是否转置
                                    OC, MB, IC,                                        // m, n, k
                                    packed_weights[i], IC,                             // a, lda
                                    src, IC,                                           // b, ldb
                                    1.0,                                               // beta
                                    dst, OC);                                          // c, ldc
            }
            //LOG(INFO) << "anakin compute[" << i << "] passed";

            if (bias_sum) {
                vsAdd(OC, bias, bias_sum, bias_sum);
                bias += OC;
            }

            // LOG(INFO) << "inputs[]:dims: " << inputs[0]->dims();
            // LOG(INFO) << "inputs:size: " << inputs.size();
            // LOG(INFO) << "inputs:capacity: " << inputs.capacity();
            // LOG(INFO) << "output:size: " << outputs.size();
            // LOG(INFO) << "OC, MB, IC: " << OC << " "<< MB << " " << IC;
        }

        if (bias) {
            #pragma omp parallel for schedule(static)
            for (cblas_int mb = 0; mb < MB; mb++) {
                cblas_saxpy(OC, 1.0, bias_sum ? bias_sum : bias, 1.0, dst + mb * OC, 1);
            }
        }
    } else {
        LOG(ERROR) << "non fp32 fc is not implemented yet";
        return SaberUnImplError;
    }

    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
