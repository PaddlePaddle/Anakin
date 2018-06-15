#ifndef ANAKIN_TEST_SABER_TEST_SABER_FUNC_BM_H
#define ANAKIN_TEST_SABER_TEST_SABER_FUNC_BM_H

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

class TestSaberFuncBM : public Test {
public:
    TestSaberFuncBM() {}
    ~TestSaberFuncBM() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}

};

#endif //ANAKIN_TEST_SABER_TEST_SABER_TENSOR_BM_H
