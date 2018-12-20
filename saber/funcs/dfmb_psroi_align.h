/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef ANAKIN_SABER_FUNCS_DFMB_PSROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_DFMB_PSROI_ALIGN_H
#include "saber/core/tensor.h"
#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_dfmb_psroi_algin.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_dfmb_psroi_align.h"
#endif
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_dfmb_psroi_algin.h"
#endif
#ifdef USE_ARM_PLACE
//todo
#include "saber/funcs/impl/impl_dfmb_psroi_algin.h"
#endif
namespace anakin {
namespace saber {
template <typename TargetType,
          DataType OpDtype
          >
class DFMBPSROIAlign : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    DFMBPSROIAlignParam
    > {
public:
    typedef TargetType targetType_t;
    typedef Tensor<TargetType> DataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef DFMBPSROIAlignParam<TargetType> Param_t;

    typedef const std::vector<DataTensor*> Input_v;
    typedef std::vector<DataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;
    DFMBPSROIAlign() = default;
    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
            Param_t& param) override {

        CHECK_EQ(input[0]->channel(),
                param.output_dim * param.group_height * param.group_width);

        CHECK_EQ(input[1]->channel(), 5);
        if (input.size() >= 3) {
            CHECK_EQ(input[2]->channel() % 2, 0);
            int num_classes = input[2]->channel() / 2;
            CHECK_EQ(param.output_dim % num_classes, 0);
            CHECK_EQ(param.part_height, input[2]->height());
            CHECK_EQ(param.part_width, input[2]->width());
        }

        Shape out_shape({input[1]->num(), param.output_dim,
                         param.pooled_height, param.pooled_width});
        return output[0]->set_shape(out_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            //                this->_impl.push_back(new VenderDFMBPSROIAlign <TargetType, OpDtype, inDtype, outDtype,
            //                LayOutType_op, LayOutType_in, LayOutType_out>);
            LOG(INFO) << "!! not impl vender dfmb_psroi_align";
            return SaberUnImplError;

        case SABER_IMPL:
            this->_impl.push_back(new SaberDFMBPSROIAlign<TargetType, OpDtype>);
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
#endif //ANAKIN_SABER_FUNCS_DFMB_PSROI_ALIGN_H
