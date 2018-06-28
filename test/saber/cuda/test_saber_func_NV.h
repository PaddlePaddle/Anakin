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

#ifndef ANAKIN_TEST_SABER_TEST_SABER_FUNC_NV_H
#define ANAKIN_TEST_SABER_TEST_SABER_FUNC_NV_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "core/tensor.h"
#include <fstream>
#include <vector>

using namespace anakin::test;

int read_file(std::vector<float> &results, const char* file_name) {

    std::ifstream infile(file_name);
    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }
    LOG(INFO)<<"found filename: "<<file_name;
    std::string line;
    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }
    return 0;
}

class TestSaberFuncNV : public Test {
public:
    TestSaberFuncNV() {}
    ~TestSaberFuncNV() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

};

#endif //ANAKIN_TEST_SABER_TEST_SABER_TENSOR_NV_H
