/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CROP_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CROP_H

#include "saber/funcs/impl/impl_crop.h"
#include "saber/funcs/crop.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberCrop<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        CropParam<X86> >
{
public:
    
    SaberCrop()
    {}

    ~SaberCrop() {
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             CropParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    };

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               CropParam<X86> &param,
                               Context<X86> &ctx) {
        this->_ctx = &ctx;
        this->_param = &param;
   	   CHECK_EQ(param.shape.size(),4);
        if (param.axis == 1) {
            CHECK_EQ(param.offset.size(), 3);
            _c_off = param.offset[0];
            _h_off = param.offset[1];
            _w_off = param.offset[2];
            _c_end = param.shape[1]+_c_off;
            _h_end = param.shape[2]+_h_off;
            _w_end = param.shape[3]+_w_off;
        } else if (param.axis == 2) {
            CHECK_EQ(param.offset.size(), 2);
            _c_off = 0;
            _h_off = param.offset[0];
            _w_off = param.offset[1];
            _c_end = param.shape[1];
            _h_end = param.shape[2]+_h_off;
            _w_end = param.shape[3]+_w_off;
        } else if (param.axis == 3) {
            CHECK_EQ(param.offset.size(), 1);
            _c_off = 0;
            _h_off = 0;
            _w_off = param.offset[0];
            _c_end = param.shape[1];
            _h_end = param.shape[2];
            _w_end = param.shape[3]+_w_off;
        } else {
            return SaberInvalidValue;
        }
        
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 CropParam<X86> &param) override;

private:
    int _c_off;
    int _h_off;
    int _w_off;
    int _c_end;
    int _h_end;
    int _w_end;
    
};

}
}
#endif
