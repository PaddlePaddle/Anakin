#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MKL_GEMM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_MKL_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/gemm.h"
#include "saber/funcs/impl/x86/mkl_gemm_int8.h"

namespace anakin {
namespace saber {


enum MKLGemmMode : int{
    NORMAL_MKLGEMM=0,
    PACKED_MKLGEMM
};

template<typename inDtype_A,typename inDtype_B,
         typename outDtype>
class MklDnnGemm{

public:


    MklDnnGemm():_s8s8s32_handle(nullptr){};
    ~MklDnnGemm() {
        if (_weights_packed_ptr_fp32 != nullptr){
            cblas_sgemm_free(_weights_packed_ptr_fp32);
        }
        if (_s8s8s32_handle!= nullptr){
            _packed_s8s8s32_gemm.release(_s8s8s32_handle);
        }
    }

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<X86> ctx,const inDtype_B* ptr_b= nullptr,MKLGemmMode gemm_mode = PACKED_MKLGEMM);

    SaberStatus dispatch(const float alpha, const float beta,int m,
                         const inDtype_A* a, const inDtype_B* b,
                         outDtype* c);

private:
    MKLGemmMode _gemm_mode{NORMAL_MKLGEMM};
    float* _weights_packed_ptr_fp32{nullptr};
    int _m{-1};
    int _n{-1};
    int _k{-1};
    int _lda{-1};
    int _ldb{-1};
    int _ldc{-1};
    float _alpha{1.f};
    float _beta{0.f};
    char _trans_a{'N'};
    char _trans_b{'N'};
    char _b_pack{'T'};
    char _offset_c_flag{'F'};
    int8_t _offset_a{0};
    int8_t _offset_b{0};
    int32_t _offset_c{0};

    MKLGEMM<AK_INT8> _packed_s8s8s32_gemm;
    Tensor<X86> _oc_offset;
    void* _s8s8s32_handle{nullptr};


};


}
}

#endif