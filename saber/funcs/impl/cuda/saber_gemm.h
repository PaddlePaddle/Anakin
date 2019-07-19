
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GEMM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GEMM_H

#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template<typename inDtype,
        typename outDtype>
class Gemm<NV, SABER_IMPL, inDtype, outDtype>
        : public MatrixFunc<NV, inDtype, outDtype>{
public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<NV> ctx);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c);

private:
    int _m{-1};
    int _n{-1};
    int _k{-1};
    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _kernel;
    Context<NV> _ctx;
};

}
}

#endif
