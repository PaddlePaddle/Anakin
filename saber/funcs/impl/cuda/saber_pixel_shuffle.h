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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PIXEL_SHUFFLE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PIXEL_SHUFFLE_H

#include "saber/funcs/impl/impl_pixel_shuffle.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPixelShuffle<NV, OpDtype>:\
    public ImplBase<
        NV,
        OpDtype,
        PixelShuffleParam<NV>> {

public:

    SaberPixelShuffle() {}

    ~SaberPixelShuffle() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             PixelShuffleParam<NV> &param,
                             Context<NV> &ctx){

      return create(inputs, outputs, param, ctx);
    }
    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               PixelShuffleParam<NV> &param,
                               Context<NV> &ctx){
      this -> _ctx = &ctx;

      _axes = inputs[0]->valid_shape().size() + 2;
      Shape in_sh = inputs[0]->valid_shape();
      int new_c = in_sh.channel()/(param.rw * param.rh);
      Shape in_new_sh;
      Shape out_new_sh;
      std::vector<int> order;
      in_new_sh.push_back(in_sh.num());
      out_new_sh.push_back(in_sh.num());
      if (param.channel_first){
        in_new_sh.push_back(new_c);
        in_new_sh.push_back(param.rh);
        in_new_sh.push_back(param.rw);
        in_new_sh.push_back(in_sh.height());
        in_new_sh.push_back(in_sh.width());
        order = std::vector<int>({0, 1, 4, 2, 5, 3});
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
        order = std::vector<int>({0, 1, 3, 2, 4, 5});  
        out_new_sh.push_back(in_sh.height());
        out_new_sh.push_back(param.rh);
        out_new_sh.push_back(in_sh.width());
        out_new_sh.push_back(param.rw); 
        out_new_sh.push_back(new_c);
      }
      Shape in_step = in_new_sh.get_stride();
      Shape out_step = out_new_sh.get_stride();

      _permute_order.reshape(Shape({6, 1, 1, 1}));
      _in_step.reshape(Shape({in_step.dims(), 1, 1, 1}));
      _out_step.reshape(Shape({out_step.dims(), 1, 1, 1}));

      cudaMemcpy(_permute_order.mutable_data(), order.data(),
                   sizeof(int) * order.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(_in_step.mutable_data(), in_step.data(),
                   sizeof(int) * _in_step.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(_out_step.mutable_data(), out_step.data(),
                   sizeof(int) * _out_step.size(), cudaMemcpyHostToDevice);
        
      return SaberSuccess;
    }
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 PixelShuffleParam<NV> &param);

private:
    int _axes;
    Tensor<NV> _permute_order;
    Tensor<NV> _in_step;
    Tensor<NV> _out_step;
};

template class SaberPixelShuffle<NV, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PixelShuffle_H
