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

#ifndef ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H
#define ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H

#include "saber/core/context.h"

namespace anakin {
namespace saber {

template <typename inTensor, 
    typename outTensor, 
    typename opTensor,
    typename Param>
class ImplBase {
public:
    typedef typename inTensor::targetType_t targetType_t;
    //typedef typename inTensor::target_type in_target;
    //typedef typename outTensor::target_type out_target;

    ImplBase() {
    }

    virtual ~ImplBase(){
	}

    virtual SaberStatus init(const std::vector<inTensor*>& inputs,
              std::vector<outTensor*>& outputs,
              Param &param, Context<targetType_t > &ctx) {
      return SaberUnImplError;
    }

    virtual SaberStatus create(const std::vector<inTensor*>& inputs,
                std::vector<outTensor*>& outputs,
                Param &param, Context<targetType_t> &ctx) {
      return SaberUnImplError;
    }

    virtual SaberStatus dispatch(const std::vector<inTensor*>& inputs,
                  std::vector<outTensor*>& outputs,
                  Param &param) {
      return SaberUnImplError;
    }

protected:
    Param* _param;
    Context<targetType_t>* _ctx;
};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H
