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

#ifndef     ANAKIN_NET_CALIBRATOR_FACTORY_H
#define    ANAKIN_NET_CALIBRATOR_FACTORY_H

#include "framework/core/net/calibrator_parse.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "framework/core/types.h"
#include <string>

namespace anakin{

OperatorBase* create_op_with_pt(std::string op_name, std::string precision, std::string target);

template <typename Target>
OperatorBase* create_precision_op(std::string op_name, std::string precision){
    LOG(INFO) << "creating op:" << op_name << "( precision:" << precision << ")";
    if (precision == "fp32"){
        return OpFactory<Target, Precision::FP32>::Global()[op_name];
    }
    if (precision == "int8"){
        return OpFactory<Target, Precision::INT8>::Global()[op_name];
    }
    LOG(FATAL) << "unsupport precision! (opname: " << op_name << ", precision:" << precision << ")";
    return nullptr;
}
template <typename Target>
OperatorBase* calibrator_op(std::string op_name, std::string name, const CalibratorParser& parser){
    std::string prec = parser.get_precision(name);
    std::string target = parser.get_target(name);
    //return create_op_with_pt(op_name, prec, target);
    //now we only support different precision
    return create_precision_op<Target>(op_name, prec);
};

}

#endif
