#ifndef ANAKIN_SABER_FUNCS_BATCH_NORM_H
#define ANAKIN_SABER_FUNCS_BATCH_NORM_H

#include "saber/core/tensor.h"
#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_batch_norm.h"

#ifdef NVIDIA_GPU
//todo
#include "saber/funcs/impl/impl_batch_norm.h"
#endif

#ifdef USE_X86_PLACE
//todo
#include "saber/funcs/impl/impl_batch_norm.h"
#endif

#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_batch_norm.h"
#endif

#ifdef USE_BM
#include "saber/funcs/impl/bm/vender_batch_norm.h"
#endif

namespace anakin {
namespace saber {

#ifdef USE_BM
template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_BM,
        DataType outDtype = AK_BM,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW
>
#else
template <typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW
>
#endif
class BatchNorm : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        BatchNormParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            BatchNormParam>::BaseFunc;

    BatchNorm() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef BatchNormParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {

        Shape output_shape = (input[0]->valid_shape());
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderBatchNorm <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                return SaberUnImplError;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace saber
} // namespace anakin

#endif
