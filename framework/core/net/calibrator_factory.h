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

#include <string>

#include "framework/core/operator/operator.h"
#include "framework/core/net/calibrator_parse.h"
#include "utils/logger/logger.h"
#include "framework/core/types.h"

namespace anakin{

OperatorBase* create_op_with_pt(std::string op_name, std::string precision, std::string target);
    
auto CalibratorOp = [](std::string op_name, std::string name, const CalibratorParser& parser){
    std::string prec = parser.get_precision(name);
    std::string target = parser.get_target(name);
    return create_op_with_pt(op_name, prec, target);
};

}

#endif
