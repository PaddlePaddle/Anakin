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

#ifndef ANAKIN_OPERATOR_UTILS_H
#define ANAKIN_OPERATOR_UTILS_H

#include "framework/core/base.h"
#include "framework/core/any.h"
#include "framework/core/data_types.h"
#include "framework/core/types.h"

namespace anakin {

template<EnumReqType ReqT>
struct EnumReqTyptify {
    typedef void type;
    const std::string info = TypeWarpper<type>().type_str;
};

/// OFFSET map to tensor type.
/// REQUEST_TYPE_WARP(OFFSET, tensor);
#define REQUEST_TYPE_WARP(EnumRequestType, RealType) \
template<>\
struct EnumReqTyptify<EnumRequestType> {\
    typedef RealType type;\
    const std::string  info = type_id<type>::type_info();\
}



/// Request type class.
template<EnumReqType ReqT>
class Request {
public:
    typedef typename EnumReqTyptify<ReqT>::type type;
    Request(type& target):_req(target) {} 

    ~Request() {}

    //! get inner data by detected type.
    auto get_data() -> type {
        return any_cast<type>(_req);
    }

private:
    any& _req;
};

} /* namespace anakin */


#endif
