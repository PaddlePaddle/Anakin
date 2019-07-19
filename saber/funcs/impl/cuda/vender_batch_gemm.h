
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_BATCH_GEMM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_BATCH_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/batch_gemm.h"

namespace anakin {
namespace saber {

template<typename inDtype,
        typename outDtype>
class BatchGemm<NV, VENDER_IMPL, inDtype, outDtype>
        : public BatchMatrixFunc<NV, inDtype, outDtype>{

public:
    BatchGemm() = default;
    ~BatchGemm() {
        if (_A != nullptr) {
            cudaFree(_A);
        }
    }

    SaberStatus init(const bool trans_a, const bool trans_b, const int max_batch,
                     Context<NV> ctx);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a[], const inDtype* b[],
                         const int m, const int n, const int k,
                         outDtype* c[], const int batch);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         inDtype* a[],
                         const int m, const int n, const int k,
                         const int batch);

private:
    Context<NV> _ctx;
    cublasHandle_t _handle{nullptr};
    cublasOperation_t _cu_trans_a;
    cublasOperation_t _cu_trans_b;
    int _m{-1};
    int _n{-1};
    int _k{-1};
    int _lda{-1};
    int _ldb{-1};
    int _ldc{-1};
    bool _trans_a;
    bool _trans_b;
    float** _A;
    float** _B;
    float** _C;
};

}
}

#endif
