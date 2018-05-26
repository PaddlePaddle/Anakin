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
#ifndef ANAKIN_TEST_SABER_TEST_SABER_FUNC_FC_X86_H
#define ANAKIN_TEST_SABER_TEST_SABER_FUNC_FC_X86_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "core/tensor.h"

using namespace anakin::test;

class TestSaberFuncFcX86 : public Test {
public:
    TestSaberFuncFcX86() {}
    ~TestSaberFuncFcX86() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}
};

#endif // ANAKIN_TEST_SABER_TEST_SABER_FUNC_FC_X86_H
