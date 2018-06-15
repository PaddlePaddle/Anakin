#ifndef ANAKIN_TEST_SABER_TEST_SABER_SHAPE_BM_H
#define ANAKIN_TEST_SABER_TEST_SABER_SHAPE_BM_H

#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "saber/core/shape.h"

using namespace anakin::test;

class TestSaberShapeBM : public Test {
public:
    TestSaberShapeBM() {}
    ~TestSaberShapeBM() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

protected:
    std::string name;
    std::string _test;
};

#endif //ANAKIN_TEST_SABER_TEST_SABER_SHAPE_BM_H

