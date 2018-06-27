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

#ifndef ANAKIN2_SABER_TEST_TEST_SABER_CONTEXT_ARM_H
#define ANAKIN2_SABER_TEST_TEST_SABER_CONTEXT_ARM_H
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "core/device.h"
#include "core/env.h"
#include "core/context.h"

using namespace anakin::test;

class TestSaberContextARM : public Test {
public:
    TestSaberContextARM() {}
    ~TestSaberContextARM() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

};


#endif //ANAKIN2_SABER_TEST_TEST_SABER_CONTEXT_ARM_H
