#include "test_saber_shape_BM.h"
#include "shape.h"
#include "anakin_config.h"

#ifdef USE_OPENMP
#include <omp.h>
#include <core/shape.h>
#endif

using namespace anakin;
using namespace saber;


TEST(TestSaberShapeBM, test_saber_shape) {

    int dim = 4;
    Shape sh4d0{0, 0, 0, 0};
    CHECK_EQ(sh4d0.dims(), 4) << "check shape dim error";

    for (int i = 0; i < dim; ++i) {
        CHECK_EQ(sh4d0[i], 0) << "check default constructor, dim size error";
    }

    CHECK_EQ(sh4d0.count(), 0) << "check shape count error";

    int N = 1;
    int C = 3;
    int H = 11;
    int W = 11;
    std::vector<int> sh_size = {N, C, H, W};
    //Shape sh4d1(sh_size);
    Shape sh4d1(N, C, H, W);
    LOG(INFO) << "Test Saber Shape, size of shape: " << sh4d1.size();
    CHECK_EQ(sh4d1.count(), N * C * H * W) << "size error with vector constructor!";
    //CHECK_EQ(sh4d2.size(), N * C * H * W) << "size error with args constructor!";

    CHECK_EQ(sh4d1[0], N) << "get shape size error";
    CHECK_EQ(sh4d1[1], C) << "get shape size error";
    CHECK_EQ(sh4d1[2], H) << "get shape size error";
    CHECK_EQ(sh4d1[3], W) << "get shape size error";

    //CHECK_EQ(sh4d2[0], N) << "get shape size error";
    //CHECK_EQ(sh4d2[1], C) << "get shape size error";
    //CHECK_EQ(sh4d2[2], H) << "get shape size error";
    //CHECK_EQ(sh4d2[3], W) << "get shape size error";

    CHECK_EQ(sh4d1.count(0), N * C * H * W) << "calculate count failed";

    C = 10;
    sh4d1[1] = C;
    CHECK_EQ(sh4d1[1], C) << "set shape size error";

    bool is_equal = (sh4d0 == sh4d1);
    CHECK_EQ(is_equal, false) << "check shape is_equal failed";

    sh4d0 = sh4d1;
    CHECK_EQ(sh4d1[0], N) << "constructor failed";
    CHECK_EQ(sh4d1[1], C) << "get shape size error";
    CHECK_EQ(sh4d1[2], H) << "get shape size error";
    CHECK_EQ(sh4d1[3], W) << "get shape size error";

    Shape sh4d3 = sh4d1;
    CHECK_EQ((sh4d3 == sh4d1), true) << "constructor error";

    Shape sh4d4(sh4d1);
    CHECK_EQ((sh4d4 == sh4d1), true) << "constructor error";

    Shape sh1d0{0};
    //std::vector<int> sh1d_size = {W};

    //Shape sh1d1(sh1d_size);
    //Shape sh1d0{W};
    Shape sh1d1(W);

    Shape sh1d3 = sh1d1;
    Shape sh1d4(sh1d1);

    CHECK_EQ(sh1d0.dims(), 1) << "shape dim error";

    CHECK_EQ(sh1d0.count(), 0) << "shape size error";

    CHECK_EQ(sh1d0.count(0), 0) << "shape1d count error";

    CHECK_EQ(sh1d1[0], W) << "get shape size error";

    //CHECK_EQ(sh1d2.count(0), W) << "shape dim error";

    CHECK_EQ((sh1d0 != sh1d1), true) << "compare shape error";

    CHECK_EQ((sh1d3 == sh1d1), true) << "compare shape error";

    CHECK_EQ((sh1d4 == sh1d1), true) << "compare shape error";

    Shape sh0{2, 2, 3, 4};
    Shape sh1{2, 1, 1, 24};
    Shape sh2{2, 2, 3, 4};
    Shape sh3{1, 1, 2, 3};

    CHECK_EQ(sh0 == sh2, true) << "error ==";
    CHECK_EQ(sh3 < sh0, true) << "error <";
    CHECK_EQ(sh3 >= sh0, false) << "error >=";
    CHECK_EQ(sh3 > sh0, false) << "error >";
    CHECK_EQ(sh0 > sh3, true) << "error >";
    CHECK_EQ(sh0 < sh1, false) << "error <";
    CHECK_EQ(sh0 <= sh2, true) << "error <=";
    CHECK_EQ(sh0 >= sh2, true) << "error >=";

    Shape sh001 = Shape::zero(2);
    Shape sh002 = Shape::zero(3);

    if (sh001 > sh002) {
        LOG(ERROR) << "error <";
    }

}


int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


