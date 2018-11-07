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

#ifndef FRAMEWORK_CORE_NET_CALIBRATOR_PARSE_H
#define FRAMEWORK_CORE_NET_CALIBRATOR_PARSE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include "utils/logger/logger.h"
#include "framework/core/types.h"
#include "saber/saber_types.h"
#include "framework/graph/graph.h"

namespace anakin{
class CalibratorParser{
public:
        CalibratorParser() = default;
        ~CalibratorParser() = default;
        void parse_from_file(std::string config, std::string calibrator);
        static void auto_config(const std::vector<std::string>& exe_nodes, const std::vector<std::string>& op_names ,std::string dst);
        std::string get_precision(std::string name) const;
        saber::DataType get_dtype(std::string name0, std::string name1) const;
        std::string get_target(std::string name) const;
        saber::LayoutType get_layout(std::string name0, std::string name1, saber::LayoutType old_layout) const;
        float get_calibrator(std::string edge_name) const;
private:
        std::unordered_map<std::string, std::string> _node_precision_map;
        std::unordered_map<std::string, std::string> _node_target_map;
        std::unordered_map<std::string, float> _node_calibrator_map;
private:
        void  _config_parse(std::string);
        void  _calibrator_parse(std::string);
        std::vector<std::string> _line_config_parse(std::string);
        void _line_calibrator_parse(std::string);
};

}

#endif
