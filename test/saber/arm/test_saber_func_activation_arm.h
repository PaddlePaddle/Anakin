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

#ifndef ANAKIN_TEST_SABER_TEST_SABER_ACTIVATION_ARM_H
#define ANAKIN_TEST_SABER_TEST_SABER_ACTIVATION_ARM_H
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include "saber/funcs/impl/arm/saber_activation.h"
using namespace anakin::test;

template <typename T>
bool compare_tensor(T& data, T& ref_data, float eps = 1e-4) {
    typedef typename T::Dtype data_t;
    if (data.size() != ref_data.size()) {
        return false;
    }
    data_t absdiff = 0.f;
    data_t absref = 0.f;
    for (int i = 0; i < data.size(); i++) {
        absdiff = std::fabs(data.data()[i] - ref_data.data()[i]);
        absref = std::fabs(ref_data.data()[i]);
        float e = absdiff > eps ? absdiff / absref : absdiff;
        if (e > eps) {
            LOG(ERROR) << "out = " << data.data()[i];
            LOG(ERROR) << "out_ref = " << ref_data.data()[i];
            return false;
        }
    }
    return true;
}

class TestSaberActivationARM : public Test {
public:
    TestSaberActivationARM() {}
    ~TestSaberActivationARM() {}
    
protected:
    virtual void setup() {}
    virtual void teardown() {}
    
};

#endif //ANAKIN_TEST_SABER_TEST_SABER_ACTIVATION_ARM_H
