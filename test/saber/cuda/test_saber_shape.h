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

#ifndef ANAKIN_TEST_SABER_TEST_ARM_TENSOR_H
#define ANAKIN_TEST_SABER_TEST_ARM_TENSOR_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "saber/core/shape.h"

using namespace anakin::test;

class TestSaberShape : public Test {
public:
    TestSaberShape() {}
    ~TestSaberShape() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

protected:
    std::string name;
    std::string _test;
};

#endif //ANAKIN_TEST_SABER_TEST_ARM_TENSOR_H

