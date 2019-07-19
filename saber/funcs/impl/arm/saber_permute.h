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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PERMUTE_H

#include "saber/funcs/impl/impl_permute.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPermute<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        PermuteParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberPermute()
    {}

    ~SaberPermute() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            PermuteParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            PermuteParam<ARM>& param, Context<ARM> &ctx) {
      _num_axes = inputs[0]->dims();
      _count = outputs[0]->valid_size();

      CHECK_EQ(inputs[0]->dims(), param.order.size()) << "ERROR: permute order size is not match to input dims\n";
      // set _need_permute
      _need_permute = false;
      for (int i = 0; i < _num_axes; ++i) {
          if (param.order[i] != i) {
              _need_permute = true;
              break;
          }
      }
      if (!_need_permute) {
          return SaberSuccess;
      }
      //! for basic permute
      std::vector<int> axis_diff;
      int j = 0;
      for (int i = 0; i < _num_axes; ++i) {
          if (param.order[j] != i) {
              axis_diff.push_back(j);
              //LOG(INFO) << "diff axis: " << _order_dims[j];
          } else {
              j++;
          }
      }

      if (inputs[0]->count_valid(axis_diff[0], _num_axes) == 1) {
          _need_permute = false;
          return SaberSuccess;
      }

      if (axis_diff.size() == 1) {
          _transpose = true;
          _trans_num = inputs[0]->count_valid(0, std::max(axis_diff[0], 0));
          _trans_w = inputs[0]->count_valid(axis_diff[0] + 1, _num_axes);
          _trans_h = inputs[0]->valid_shape()[axis_diff[0]];
  #ifdef ENABLE_DEBUG
          printf("permute: transpose=true, num= %d, h=%d, w=%d\n", _trans_num , _trans_h, _trans_w);
  #endif
      } else {
          _transpose = false;
          _new_steps = outputs[0]->get_stride();
          _old_steps = inputs[0]->get_stride();
  #ifdef ENABLE_DEBUG
          printf("permute: transpose=false\n");
  #endif
      }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          PermuteParam<ARM>& param);
private:
    int _num_axes;
    int _count;
    bool _need_permute{false};
    bool _transpose{false};
    int _trans_num;
    int _trans_w;
    int _trans_h;
    std::vector<int> _new_steps;
    std::vector<int> _old_steps;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Permute_H
