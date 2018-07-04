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

#ifndef ANAKIN2_TEST_SABER_TEST_SABER_FUNC_TEST_ARM_H
#define ANAKIN2_TEST_SABER_TEST_SABER_FUNC_TEST_ARM_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include <fstream>
#include <vector>

using namespace anakin::test;

int read_file(std::vector<float> &results, const char* file_name) {

    std::ifstream infile(file_name);
    if (!infile.good()) {
        LOG(ERROR) << "Cannot open " << file_name;
        return false;
    }
    LOG(INFO) << "found filename: " << file_name;
    std::string line;
    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }
    return 0;
}

class TestSaberLite : public Test {
public:
    TestSaberLite() {}
    ~TestSaberLite() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

};

#endif //ANAKIN2_TEST_SABER_TEST_SABER_FUNC_TEST_ARM_H
