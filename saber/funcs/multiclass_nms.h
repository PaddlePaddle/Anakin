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

#ifndef ANAKIN_SABER_FUNCS_MULTICLASS_NMS_H
#define ANAKIN_SABER_FUNCS_MULTICLASS_NMS_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_multiclass_nms.h"
#endif

#ifdef USE_X86_PLACE
//#include "saber/funcs/impl/x86/saber_activation.h"
#endif

namespace anakin{

namespace saber{

template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NHW,
        typename LayOutType_in = NHW,
        typename LayOutType_out = NW
>
class MultiClassNMS : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        MultiClassNMSParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            MultiClassNMSParam>::BaseFunc;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef MultiClassNMSParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    MultiClassNMS() = default;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        //! inputs[0]: bbox map, dims = 3 {N, boxes, 4(xmin, ymin, xmax, ymax)}
        //! inputs[1]: score map, dims = 3 {N, classes, boxes}
        //! output[0]: output detection result, dims = 2 {No., 6}
        Shape sh1 = input[0]->valid_shape();
        Shape sh2 = input[1]->valid_shape();
        CHECK_EQ(sh1.dims(), 3) << "only support 3d (NHW) layout";
        Shape shape_out = output[0]->valid_shape();
        CHECK_EQ(shape_out.dims(), 2) << "only support 2d(NW) layout";
        int boxes = sh1[1];
        shape_out[0] = 1;
        shape_out[1] = 7;
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMultiClassNMS <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberMultiClassNMS <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            default:
                return SaberUnImplError;            
        }
    }

private:

    virtual void pick_best_static() override {
        //! Fc only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_runtime(Input_v input, Output_v output, \
        Param_t& param, Context<TargetType> &ctx) override {
        //! Fc only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //! Fc only has saber implementation
        this->_best_impl = this->_impl[0];
    }

};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_MULTICLASS_NMS_H
