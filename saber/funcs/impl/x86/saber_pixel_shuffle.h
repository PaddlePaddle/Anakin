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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PIXEL_SHUFFLE_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PIXEL_SHUFFLE_H

#include "saber/funcs/impl/impl_pixel_shuffle.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPixelShuffle<X86, OpDtype>:\
    public ImplBase<
        X86,
        OpDtype,
        PixelShuffleParam<X86>> {

public:

    SaberPixelShuffle() {}
    ~SaberPixelShuffle() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             PixelShuffleParam<X86> &param,
                             Context<X86> &ctx){
      return create(inputs, outputs, param, ctx);
    }
    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               PixelShuffleParam<X86> &param,
                               Context<X86> &ctx){
      this -> _ctx = &ctx;

      _num_axes = inputs[0]->valid_shape().size() + 2;
      Shape in_sh = inputs[0]->valid_shape();
      int new_c = in_sh.channel()/(param.rw * param.rh);
      Shape in_new_sh;
      Shape out_new_sh;
      in_new_sh.push_back(in_sh.num());
      out_new_sh.push_back(in_sh.num());
      if (param.channel_first){
        in_new_sh.push_back(new_c);
        in_new_sh.push_back(param.rh);
        in_new_sh.push_back(param.rw);
        in_new_sh.push_back(in_sh.height());
        in_new_sh.push_back(in_sh.width());
        _order = std::vector<int>({0, 1, 4, 2, 5, 3});
        out_new_sh.push_back(new_c);
        out_new_sh.push_back(in_sh.height());
        out_new_sh.push_back(param.rh);
        out_new_sh.push_back(in_sh.width());
        out_new_sh.push_back(param.rw);
        

      } else {
        in_new_sh.push_back(in_sh.height());
        in_new_sh.push_back(in_sh.width());
        in_new_sh.push_back(param.rh);
        in_new_sh.push_back(param.rw);
        in_new_sh.push_back(new_c);
        _order = std::vector<int>({0, 1, 3, 2, 4, 5}); 
        out_new_sh.push_back(in_sh.height());
        out_new_sh.push_back(param.rh);
        out_new_sh.push_back(in_sh.width());
        out_new_sh.push_back(param.rw); 
        out_new_sh.push_back(new_c);
      }
      _in_steps = in_new_sh.get_stride();
      _out_steps = out_new_sh.get_stride();

        
      return SaberSuccess;
    }
    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 PixelShuffleParam<X86> &param);

private:
    int _num_axes;
    std::vector<int> _order;
    Shape _in_steps;
    Shape _out_steps;
    Shape _out_new_sh;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PixelShuffle_H
