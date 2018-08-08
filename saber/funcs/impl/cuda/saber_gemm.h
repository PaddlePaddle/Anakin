
#ifndef SABER_FUNCS_IMPL_CUDA_SABER_GEMM_H
#define SABER_FUNCS_IMPL_CUDA_SABER_GEMM_H

#include "saber/funcs/gemm.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
namespace anakin {
namespace saber {

template<typename Dtype>
class Gemm<NV, SABER_IMPL, Dtype> {
public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<NV> ctx);

    SaberStatus dispatch(const Dtype alpha, const Dtype* ptr_a,
                         const Dtype beta, const Dtype* ptr_b, Dtype* ptr_c) {

        cudaStream_t cuda_stream = _ctx.get_compute_stream();
        _kernel(_m, _n, _k, alpha, ptr_a, beta, ptr_b, ptr_c, cuda_stream);
        return SaberSuccess;
    }

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