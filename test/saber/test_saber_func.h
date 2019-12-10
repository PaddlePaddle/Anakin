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

#ifndef ANAKIN_TEST_SABER_FUNC_H
#define ANAKIN_TEST_SABER_FUNC_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "core/tensor.h"
#include <fstream>
#include <vector>
#include <cmath>

using namespace anakin::test;

static void split_string(const std::string& s, char delim,
                         std::vector<std::string>& elems) {
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}
template <typename Dtype>
void tensor_cmp_host_mlu(const Dtype* src1, const Dtype* src2, \
                         int size, double& max_ratio, double& max_diff) {
    double sum_diff_sq = 0.0;
    double sum_x_sq = 0.0;
    double eps = 1e-10;
    for (size_t i = 0; i < size; i++) {
        if (std::isnan(src1[i]) || std::isnan(src2[2])){
            max_ratio = 9999;
            max_diff = 9999;
            return;
        }
        sum_diff_sq += (src1[i] - src2[i]) * (src1[i] - src2[i]);
        sum_x_sq += src2[i] * src2[i];
    }

    max_ratio = sqrt(sum_diff_sq / (sum_x_sq + eps));
    max_diff = max_ratio;
}

bool read_file(std::vector<float>& results, const char* file_name, char split_char, int index) {

    std::ifstream infile(file_name);

    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }

    LOG(INFO) << "found filename: " << file_name;
    std::string line;

    while (std::getline(infile, line)) {
        std::vector<std::string> vec;
        split_string(line, split_char, vec);
        results.push_back((float)atof(vec[index].c_str()));
    }

    return true;
}
bool read_file(std::vector<float>& results, const char* file_name) {

    std::ifstream infile(file_name);

    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }

    LOG(INFO) << "found filename: " << file_name;
    std::string line;

    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }

    return true;
}
class TestSaberFunc : public Test {
public:
    TestSaberFunc() {}
    ~TestSaberFunc() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

};

#endif //ANAKIN_TEST_SABER_TEST_SABER_TENSOR_NV_H
