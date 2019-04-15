#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_H
#include <memory>

#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_deconv_act_kernel.h"

namespace anakin {
namespace saber {

using namespace jit;

template<DataType OpDtype = AK_FLOAT>
class JitAvx2Deconv : public ImplBase<
        X86, OpDtype, ConvParam <X86>> {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    JitAvx2Deconv() : kernel(nullptr) {}
    ~JitAvx2Deconv() {
        if (kernel) {
            delete kernel;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             ConvParam<X86> &param, Context<X86>&ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               ConvParam<X86> &param, Context<X86>&ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor < X86>*>& outputs,
                                 ConvParam <X86> &param) override;
private:
    jit_deconv_conf_t conf;
    jit_avx2_deconv_act_kernel *kernel = nullptr;
    std::shared_ptr<Tensor<X86> > weights_internal;
    std::shared_ptr<Tensor<X86> > bias_internal;
    SaberStatus check_conf(const std::vector<Tensor<X86> *>& inputs,
                        std::vector<Tensor<X86>*>& outputs,
                        ConvParam<X86> &param);
};

}
}
#endif // ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_H