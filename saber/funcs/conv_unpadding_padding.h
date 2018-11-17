#ifndef ANAKIN_SABER_FUNCS_CONV_UNPADDING_PADDING_H
#define ANAKIN_SABER_FUNCS_CONV_UNPADDING_PADDING_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/impl_conv_unpadding_padding.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_conv_upadding_padding.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
         DataType OpDtype>
class ConvUnpaddingPadding : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    ConvUnpaddingPaddingParam > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    ConvUnpaddingPaddingParam >::BaseFunc;

    ConvUnpaddingPadding() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ConvUnpaddingPaddingParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
            Output_v& output, Param_t& param) override {

        Shape in_shape = input[0]->valid_shape();
        //        in_shape[2]=
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(in_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            this->_impl.push_back(new VenderConvUnpaddingPadding<TargetType,
                                  OpDtype>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberConvUnpaddingPadding<TargetType,
                                  OpDtype>);
            return SaberSuccess;

        default:
            return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};


}
}

#endif //ANAKIN_SABER_FUNCS_CONV_UNPADDING_PADDING_H
