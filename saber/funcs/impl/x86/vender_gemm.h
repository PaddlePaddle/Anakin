
#ifndef SABER_FUNCS_IMPL_X86_VENDER_GEMM_H
#define SABER_FUNCS_IMPL_X86_VENDER_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/gemm.h"
#include "mkl.h"

namespace anakin {
namespace saber {

template<typename inDtype,
        typename outDtype>
class Gemm<X86, VENDER_IMPL, inDtype, outDtype> {

public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<X86> ctx);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c);

private:
    Context<X86> _ctx;
    CBLAS_LAYOUT _layout = CblasRowMajor;
    CBLAS_TRANSPOSE c_trans_a;
    CBLAS_TRANSPOSE c_trans_b;
    int _m{-1};
    int _n{-1};
    int _k{-1};
    int _lda{-1};
    int _ldb{-1};
    int _ldc{-1};
};

}
}

#endif