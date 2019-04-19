#ifndef ANAKIN_SABER_FUNCS_BOX_CLIP_H
#define ANAKIN_SABER_FUNCS_BOX_CLIP_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_box_clip.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_box_clip.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_box_clip.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
         DataType OpDtype>

class BoxClip : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    EmptyParam
    > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    EmptyParam >::BaseFunc;

    BoxClip() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef EmptyParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
            Output_v& output, Param_t& param) override {

        output[0]->set_seq_offset(input[1]->get_seq_offset());
        return output[0]->set_shape_without_layout(input[1]->valid_shape());
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            this->_impl.push_back(new VenderBoxClip <TargetType, OpDtype>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberBoxClip <TargetType, OpDtype>);
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

} // namespace saber
} // namespace anakin

#endif //ANAKIN_BOX_CLIP_H
