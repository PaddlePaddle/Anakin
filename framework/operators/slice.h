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

#ifndef ANAKIN_OPERATOR_SLICE_H
#define ANAKIN_OPERATOR_SLICE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/slice.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class SliceHelper;

/// pooling op
/**
 * \brief Slice implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Slice : public Operator<Ttype, Dtype, Ptype> {
public:
    Slice() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class SliceHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_SLICE(Ttype, Dtype, Ptype) \
template<> \
void Slice<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<SliceHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SliceHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_slice; \
    impl->_funcs_slice(ins, outs, param, ctx); \
}
/**
 * \brief Slice helper class to implement Slice
 * public inherit OperatorHelper
 * including init resource and shape size in Slice context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class SliceHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    SliceHelper()=default;

    ~SliceHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Slice op parameter.";
        auto slice_dim = GET_PARAMETER(int, slice_dim);
        _slice_point = GET_PARAMETER(PTuple<int>, slice_point);
        _axis = GET_PARAMETER(int, axis);
                LOG(INFO) << " slice_dim " << slice_dim;
                LOG(INFO) << " slice_point size(" << _slice_point.size() << ").";

        for (auto item : _slice_point.vector()) {
                    LOG(INFO) << "  |-- " << item;
        }

                LOG(INFO) << " axis " << _axis;

        SliceParam<Tensor4d<Ttype, Dtype>> param_slice(_axis, _slice_point.vector());
        _param_slice = param_slice;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Slice operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_slice.init(ins, outs, _param_slice, SPECIFY, SABER_IMPL, ctx));
        return Status::OK();
    }

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        if (_slice_point.size() + 1 != outs.size()) {
            if (_slice_point.size() == 1) {
                for (int i = 0; i < outs.size() - 2; i++) {
                    _slice_point.push_back(_slice_point[0] + _slice_point[_slice_point.size() - 1]);
                }

                SliceParam<Tensor4d<Ttype, Dtype>> param_slice(_axis, _slice_point.vector());
                _param_slice = param_slice;
            }
        }

        SABER_CHECK(_funcs_slice.compute_output_shape(ins, outs, _param_slice));
        return Status::OK();
    }

public:
    ///< _param_slice stand for slice parameter
    saber::SliceParam<Tensor4d<Ttype, Dtype>> _param_slice;
    ///< _funcs_slice stand for slice function 
    saber::Slice<Ttype, Dtype> _funcs_slice;

private:
   ///< _slice_point stand for op slice
    PTuple<int> _slice_point; 
    ///< _axis stand for axis of input to slice
    int _axis;
};

#ifdef USE_CUDA
INSTANCE_SLICE(NV, AK_FLOAT, Precision::FP32);
template class SliceHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, NV, AK_FLOAT, Precision::FP32);
template class SliceHelper<NV, AK_FLOAT, Precision::FP16>;
template class SliceHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SLICE(ARM, AK_FLOAT, Precision::FP32);
template class SliceHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Slice)
.Doc("Slice operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("slice")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("slice")
#endif
.num_in(1)
.num_out(1)
.Args<int>("slice_dim", " slice dim at input ")
.Args<PTuple<int>>("slice_point", " slice point of op")
.Args<int>("axis", " axis of input to slice");

} /* namespace ops */

} /* namespace anakin */

#endif
