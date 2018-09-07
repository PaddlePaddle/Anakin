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

#ifndef ANAKIN_OPERATOR_FUNC_H
#define ANAKIN_OPERATOR_FUNC_H

#include "framework/core/base.h"
#include "framework/core/operator/operator.h"

namespace anakin {

/** 
 *  \brief Operator executor class.
 */
template<typename Ttype, Precision Ptype>
struct OperatorFunc {
    OperatorFunc() {}

    /** 
     *  \brief Launch a operator.
     */
    void launch();

    /** 
     *  \brief Infer shape.
     */
    void infer_shape();
    
    ///< op running context.
    OpContextPtr<Ttype> ctx_p;
    
    ///< request list for operators.
    ///< std::vector<Request<EnumReqType> > requests.
    ///< input data of operator.
    std::vector<Tensor4dPtr<Ttype> > ins;

    ///< the lanes int data resides in
    std::vector<graph::Lane> in_lanes;
    
    ///< output data of operator
    std::vector<Tensor4dPtr<Ttype> > outs;
    
    ///< the lanes out data resides in
    std::vector<graph::Lane> out_lanes;

    ///< the current lane the operator execute
    graph::Lane current_lane;

    bool need_sync{false};

    Operator<Ttype, Ptype>* op;

    ///< node name
    std::string name;
    ///< operation name
    std::string op_name;
};

} /* namespace */

#endif
