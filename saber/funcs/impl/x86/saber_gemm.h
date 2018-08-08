
#ifndef SABER_FUNCS_IMPL_X86_SABER_GEMM_H
#define SABER_FUNCS_IMPL_X86_SABER_GEMM_H

#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template<typename Dtype>
class Gemm<X86, SABER_IMPL, Dtype> {
public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_A, const bool trans_B,
                     const int m, const int n, const int k,
                     Context<X86> ctx) {

        LOG(INFO) << "UNIMPLEMENT";
        return SaberUnImplError;
    }

    SaberStatus dispatch(const int m, const int n, const int k,
                         const Dtype alpha, const Dtype* a,
                         const Dtype beta, const Dtype* b,
                         Dtype* c) {

        LOG(INFO) << "UNIMPLEMENT";
        return SaberUnImplError;
    }

private:


};

}
}

#endif