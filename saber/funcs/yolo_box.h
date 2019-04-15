/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_YOLO_BOX_H
#define ANAKIN_SABER_FUNCS_YOLO_BOX_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_yolo_box.h"

#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/saber_yolo_box.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_yolo_box.h"
#endif
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_yolo_box.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class YoloBox : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        YoloBoxParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            YoloBoxParam>::BaseFunc;

    YoloBox() = default;

    virtual SaberStatus compute_output_shape(
            const std::vector<Tensor<TargetType>*>& input,
            std::vector<Tensor<TargetType>*> &output,
            YoloBoxParam<TargetType> &param) override {

        auto dim_x = input[0]->valid_shape();
        auto dim_imgsize = input[1]->valid_shape();
        auto anchors = param.anchors;
        int anchor_num = anchors.size() / 2;
        auto class_num = param.class_num;


        CHECK_EQ(dim_x[1], anchor_num * (5 + class_num))
            << "Input(X) dim[1] should be equal to (anchor_mask_number * (5 + class_num)).";
        CHECK_EQ(dim_imgsize[0], dim_x[0])
                << "Input(ImgSize) dim[0] and Input(X) dim[0] should be same.";

        CHECK_EQ(dim_imgsize[1], 2) << "Input(ImgSize) dim[1] should be 2.";
        CHECK_GT(anchors.size(), 0) << "Attr(anchors) length should be greater than 0.";
        CHECK_EQ(anchors.size() % 2, 0) << "Attr(anchors) length should be even integer.";
        CHECK_GT(class_num, 0) << "Attr(class_num) should be an integer greater than 0.";

        int box_num = dim_x[2] * dim_x[3] * anchor_num;
        Shape dim_boxes({dim_x[0], box_num, 4}, Layout_NHW);
        output[0]->set_shape(dim_boxes);

        Shape dim_scores({dim_x[0], box_num, class_num}, Layout_NHW);
        output[1]->set_shape(dim_scores);

        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderYoloBox <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberYoloBox <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace saber
} // namespace anakin

#endif
