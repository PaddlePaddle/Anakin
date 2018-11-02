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
#include "framework/core/net/calibrator_parse.h"
#include <fstream>
#include <iostream>

namespace anakin {

std::string CalibratorParser::get_precision(std::string name) const {
    //if not exist, return fp32
    if (_node_precision_map.find(name) == _node_precision_map.end()) {
        return "fp32";
    }

    return _node_precision_map.at(name);
}
saber::DataType CalibratorParser::get_dtype(std::string name0, std::string name1) const {
    std::string str0 = get_precision(name0);
    std::string str1 = get_precision(name1);
    bool bint8 = (str0 == "int8") && (str1 == "int8");

    if (!bint8) {
        return saber::AK_FLOAT;
    } else {
        return saber::AK_INT8;
    }
}

std::string CalibratorParser::get_target(std::string name) const {
    //if not exist, return NV
    if (_node_target_map.find(name) == _node_target_map.end()) {
        return "AMD";
    }

    return _node_target_map.at(name);
}
float CalibratorParser::get_calibrator(std::string name) const {
    //if not exist, return 1.0f
    if (_node_calibrator_map.find(name) == _node_calibrator_map.end()) {
        return 1.0f;
    }

    return _node_calibrator_map.at(name);
}
saber::LayoutType CalibratorParser::get_layout(std::string name0, std::string name1,
        saber::LayoutType old_layout) const {
    std::string str0 = get_precision(name0);
    std::string str1 = get_precision(name1);
    bool bint8 = (str0 == "int8") && (str1 == "int8");

    if (!bint8) {
        return old_layout;
    } else {
        return saber::Layout_NCHW_C4;
    }

}

void CalibratorParser::auto_config(const std::vector<std::string>& exe_nodes,
                                   const std::vector<std::string>& op_names, std::string dst) {
    std::fstream fs;
    fs.open(dst, std::ios::in);

    if (fs) {
        fs.close();
        LOG(WARNING) << "config file already existed, will not be created ";
        return;
    }

    LOG(WARNING) << "config file not existed, creating it ";
    std::ofstream ofs(dst);

    if (!ofs.is_open()) {
        LOG(FATAL) << "open file " << dst << "failed";
    }

    for (int i = 0; i < exe_nodes.size(); ++i) {
        std::string name = exe_nodes[i];

        if (!name.empty()) {
            std::string op_name = op_names[i];
            ofs << name << "(" << op_name << ")    fp32    NV \n";
        }
    }

    ofs.close();
}

void CalibratorParser::parse_from_file(std::string config, std::string calibrator) {
    _config_parse(config);
    _calibrator_parse(calibrator);
}

void CalibratorParser::_config_parse(std::string config) {
    std::ifstream ifs(config);

    if (!ifs.is_open()) {
        LOG(ERROR) << "open file " << config << " failed, will use default config";
        return;
    }

    std::string line;

    while (ifs.good()) {
        std::getline(ifs, line);

        if (!line.empty()) {
            auto str_vec = _line_config_parse(line);
            std::string node_name;

            if (str_vec.size() >= 1) {
                node_name = str_vec[0];
                node_name.erase(node_name.find("("));
            }

            if (str_vec.size() >= 3) {
                _node_target_map[node_name] = str_vec[2];
            }

            if (str_vec.size() >= 2) {
                _node_precision_map[node_name] = str_vec[1];
            }
        }
    }

    ifs.close();
}
void CalibratorParser::_calibrator_parse(std::string calibrator) {
    std::ifstream ifs(calibrator);

    if (!ifs.is_open()) {
        LOG(WARNING) << "open file " << calibrator << "failed!, will use default calibrator";
        return;
    }

    std::string line;

    while (ifs.good()) {
        std::getline(ifs, line);

        if (!line.empty()) {
            _line_calibrator_parse(line);
        }
    }

    ifs.close();
}

std::vector<std::string> CalibratorParser::_line_config_parse(std::string line) {
    line.erase(line.find_last_not_of("\n") + 1);
    line.erase(line.find_last_not_of(" ") + 1);
    std::istringstream iss(line);
    std::string temp;
    std::vector<std::string> str_vec;

    while (iss.good()) {
        iss >> temp;
        str_vec.push_back(temp);
    }

    return str_vec;
}

void CalibratorParser::_line_calibrator_parse(std::string line) {
    line.erase(line.find_last_not_of("\n") + 1);
    line.erase(line.find_last_not_of(" ") + 1);
    std::istringstream iss(line);
    std::string name;
    float value = 1.0f;

    if (iss.good()) {
        iss >> name;
    }

    try {
        if (iss.good()) {
            iss.precision(7);
            iss >> value;
        }
    } catch (std::exception& e) {
        LOG(FATAL) << "calibrator load wrong!! line:" << line;
    }

    _node_calibrator_map[name] = value;
}

}
