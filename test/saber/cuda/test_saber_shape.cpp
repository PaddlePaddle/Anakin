
#include "test_saber_shape.h"
#include "shape.h"
#include "anakin_config.h"

#ifdef USE_OPENMP
#include <omp.h>
#include <core/shape.h>

#endif

using namespace anakin;
using namespace saber;

/*
TEST(TestSaberShape, test_saber_shape) {

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
*/

bool test_dim4(LayoutType  layout_type) {
    std::vector<int> data;
    int inner_c = 1;
    std::vector<int> check;
    for (int N = 1; N < 20; ++N) {
        for (int C = 1; C < 20; ++C) {
            for (int H = 1; H < 20; ++H) {
                for (int W = 1; W < 20; ++W) {
                    switch(layout_type) {
                        case Layout_invalid:
                            inner_c = 1; data = {1, 1, 1, 1}; check = {N, C, H, W}; break;
                        case Layout_NCHW:
                            inner_c = 1; data = {N, C, H, W}; check = {N, C, H, W}; break;
                        case Layout_NHWC:
                            inner_c = 1; data = {N, H, W, C}; check = {N, H, W, C}; break;
                        case Layout_NCHW_C4:
                            inner_c = 4; data = {N, C, H, W, inner_c}; check = {N, C, H, W}; break;
                        case Layout_NCHW_C8:
                            inner_c = 8; data = {N, C, H, W, inner_c}; check = {N, C, H, W}; break;
                        case Layout_NCHW_C16:
                            inner_c = 16; data = {N, C, H, W, inner_c}; check = {N, C, H, W}; break;
                        default:
                            continue;
                    }
                    Shape sh1(data, layout_type);
                    Shape sh2;
                    sh2 = sh1;
                    bool equal_shape = sh2 == sh1;
                    Shape stride = sh1.get_stride();
                    Shape sh3 = sh2;
                    sh3 = sh1 + sh2;
                    bool flag = sh3 > sh1;
                    flag &= sh1 < sh3;
                    flag &= sh2 <= sh3;
                    flag &= sh3 >= sh1;
                    Shape zero_shape = Shape::zero(sh3);
                    Shape minus_shape = Shape::minusone(sh3);
                    CHECK_EQ(equal_shape, true);
                    CHECK_EQ(sh1.num(), N);
                    CHECK_EQ(sh1.channel(), C * inner_c);
                    CHECK_EQ(sh1.height(), H);
                    CHECK_EQ(sh1.width(), W);
                    CHECK_EQ(sh1.depth(), 1);
                    CHECK_EQ(sh1.count(), N * C * H * W * inner_c);
                    CHECK_EQ(sh1.count(1), check[1] * check[2] * check[3] * inner_c);
                    CHECK_EQ(sh1.count(2), check[2] * check[3] * inner_c);
                    CHECK_EQ(sh1.count(3), check[3] * inner_c);
                    CHECK_EQ(sh1.get_layout(), layout_type);
                    CHECK_EQ(sh1.dims(), data.size());
                    CHECK_EQ(sh2.num(), N);
                    CHECK_EQ(sh2.channel(), C * inner_c);
                    CHECK_EQ(sh2.height(), H);
                    CHECK_EQ(sh2.width(), W);
                    CHECK_EQ(sh2.depth(), 1);
                    CHECK_EQ(sh2.count(), N * C * H * W * inner_c);
                    CHECK_EQ(sh2.count(1), check[1] * check[2] * check[3] * inner_c);
                    CHECK_EQ(sh2.count(2), check[2] * check[3] * inner_c);
                    CHECK_EQ(sh2.count(3), check[3] * inner_c);
                    CHECK_EQ(sh2.count(1, 4),  check[1] * check[2] * check[3]);
                    CHECK_EQ(sh2.count(1, 2),  data[1]);
                    CHECK_EQ(sh2.count(1, 1),  1);
                    CHECK_EQ(sh2.get_layout(), layout_type);
                    CHECK_EQ(sh2.dims(), data.size());
                    CHECK_EQ(stride[0], check[1] * check[2] * check[3] * inner_c);
                    CHECK_EQ(stride[1], check[2] * check[3] * inner_c);
                    CHECK_EQ(stride[2], check[3] * inner_c);
                    CHECK_EQ(stride[3], inner_c);
                    CHECK_EQ(stride.get_layout(), layout_type);
                    CHECK_EQ(stride.dims(), data.size());
                    CHECK_EQ(flag, true);
                    CHECK_EQ(sh3.num(), 2 * N);
                    CHECK_EQ(sh3.channel(), 2 * C * inner_c);
                    CHECK_EQ(sh3.height(), 2 * H);
                    CHECK_EQ(sh3.width(), 2 * W);
                    CHECK_EQ(sh3.get_layout(), layout_type);
                    CHECK_EQ(sh3.dims(), data.size());
                    CHECK_EQ(zero_shape.num(), 0);
                    CHECK_EQ(zero_shape.channel(), 0);
                    CHECK_EQ(zero_shape.height(), 0);
                    CHECK_EQ(zero_shape.width(), 0);
                    CHECK_EQ(zero_shape.get_layout(), layout_type);
                    CHECK_EQ(zero_shape.dims(), data.size());
                    CHECK_EQ(minus_shape.num(), -1);
                    CHECK_EQ(minus_shape.channel(), -1 * inner_c);
                    CHECK_EQ(minus_shape.height(), -1);
                    CHECK_EQ(minus_shape.width(), -1);
                    CHECK_EQ(minus_shape.get_layout(), layout_type);
                    CHECK_EQ(minus_shape.dims(), data.size());
                }
            }
        }
    }
    return true;
}

bool test_dim2(LayoutType  layout_type) {
    std::vector<int> data;
    int inner_c = 1;
    std::vector<int> check;
    for (int N = 1; N < 20; ++N) {
        for (int C = 1; C < 20; ++C) {
            for (int H = 1; H < 20; ++H) {
                for (int W = 1; W < 20; ++W) {
                    switch(layout_type) {
                        case Layout_invalid:
                            inner_c = 1; data = {1, 1, 1, 1}; check = {N, C, H, W}; break;
                        case Layout_HW:
                            inner_c = 1; data = {H, W}; check = {1, 1, H, W}; break;
                        case Layout_WH:
                            inner_c = 1; data = {W, H}; check = {1, 1, H, W}; break;
                        case Layout_NW:
                            inner_c = 1; data = {N, W}; check = {N, 1, 1, W}; break;
                        default:
                            continue;
                    }
                    Shape sh1(data, layout_type);
                    Shape sh2;
                    sh2 = sh1;
                    bool equal_shape = sh2 == sh1;
                    Shape stride = sh1.get_stride();
                    Shape sh3 = sh2;
                    sh3 = sh1 + sh2;
                    bool flag = sh3 > sh1;
                    flag &= sh1 < sh3;
                    flag &= sh2 <= sh3;
                    flag &= sh3 >= sh1;
                    Shape zero_shape = Shape::zero(sh3);
                    Shape minus_shape = Shape::minusone(sh3);
                    CHECK_EQ(equal_shape, true);
                    CHECK_EQ(sh1.num(), check[0]);
                    CHECK_EQ(sh1.channel(), check[1] * inner_c);
                    CHECK_EQ(sh1.height(), check[2]);
                    CHECK_EQ(sh1.width(), check[3]);
                    CHECK_EQ(sh1.depth(), 1);
                    CHECK_EQ(sh1.count(), check[0] * check[1] * check[2] * check[3] * inner_c);
                    CHECK_EQ(sh1.count(1), check[3] * inner_c);
                    CHECK_EQ(sh1.count(2), 1);
                    CHECK_EQ(sh1.count(3), 1);
                    CHECK_EQ(sh1.get_layout(), layout_type);
                    CHECK_EQ(sh1.dims(), data.size());
                    CHECK_EQ(sh2.num(), check[0]);
                    CHECK_EQ(sh2.channel(), check[1] * inner_c);
                    CHECK_EQ(sh2.height(), check[2]);
                    CHECK_EQ(sh2.width(), check[3]);
                    CHECK_EQ(sh2.depth(), 1);
                    CHECK_EQ(sh2.count(), check[0] * check[1] * check[2] * check[3] * inner_c);
                    CHECK_EQ(sh2.count(1), check[3] * inner_c);
                    CHECK_EQ(sh2.count(2), 1);
                    CHECK_EQ(sh2.count(3), 1);
                    CHECK_EQ(sh2.get_layout(), layout_type);
                    CHECK_EQ(sh2.dims(), data.size());
                    CHECK_EQ(stride[0], check[3] * inner_c);
                    CHECK_EQ(stride[1], inner_c);
                    CHECK_EQ(stride.get_layout(), layout_type);
                    CHECK_EQ(stride.dims(), data.size());
                    CHECK_EQ(flag, true);
                    CHECK_EQ(sh3.width(), 2 * W);
                    CHECK_EQ(sh3.get_layout(), layout_type);
                    CHECK_EQ(sh3.dims(), data.size());
                    CHECK_EQ(zero_shape.width(), 0);
                    CHECK_EQ(zero_shape.get_layout(), layout_type);
                    CHECK_EQ(zero_shape.dims(), data.size());
                    CHECK_EQ(minus_shape.width(), -1);
                    CHECK_EQ(minus_shape.get_layout(), layout_type);
                    CHECK_EQ(minus_shape.dims(), data.size());
                }
            }
        }
    }
    return true;
}

TEST(TestSaberShape, test_dim_4) {
    // constructor test
    test_dim4(Layout_NCHW);
    LOG(INFO) << "Layout_NCHW PASS";
    test_dim4(Layout_NHWC);
    LOG(INFO) << "Layout_NHWC PASS";
    test_dim4(Layout_NCHW_C4);
    LOG(INFO) << "Layout_NCHW_C4 PASS";
    test_dim4(Layout_NCHW_C8);
    LOG(INFO) << "Layout_NCHW_C8 PASS";
    test_dim4(Layout_NCHW_C16);
    LOG(INFO) << "Layout_NCHW_C16 PASS";
}

TEST(TestSaberShape, test_dim_2) {
    // constructor test
    test_dim2(Layout_NW);
            LOG(INFO) << "Layout_NW PASS";
    test_dim2(Layout_HW);
            LOG(INFO) << "Layout_HW PASS";
}

TEST(TestSaberShape, test_set_layout) {
    Shape test_shape;
    test_shape.push_back(2);
    test_shape.push_back(8);
    test_shape.push_back(4);
    test_shape.push_back(5);

    test_shape.set_layout(Layout_NCHW);
    LOG(INFO) <<"NCHW";
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
    LOG(INFO) << "test_shape[2] = "<< test_shape[2];
    LOG(INFO) << "test_shape[3] = "<< test_shape[3];
    test_shape.set_layout(Layout_NHWC);
    LOG(INFO) <<"NHWC";
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
    LOG(INFO) << "test_shape[2] = "<< test_shape[2];
    LOG(INFO) << "test_shape[3] = "<< test_shape[3];
    test_shape.set_layout(Layout_NCHW_C4);
    LOG(INFO) <<"NCHW_C4";
    LOG(INFO) << "test_shape.channel = "<< test_shape.channel();
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
    LOG(INFO) << "test_shape[2] = "<< test_shape[2];
    LOG(INFO) << "test_shape[3] = "<< test_shape[3];
    test_shape.set_layout(Layout_HW);
    LOG(INFO) <<"HW";
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
    test_shape.set_layout(Layout_NCHW);
    LOG(INFO) <<"NCHW";
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
    LOG(INFO) << "test_shape[2] = "<< test_shape[2];
    LOG(INFO) << "test_shape[3] = "<< test_shape[3];
    test_shape.set_layout(Layout_HW, {10, 5});
    LOG(INFO) <<"HW";
    LOG(INFO) << "test_shape[0] = "<< test_shape[0];
    LOG(INFO) << "test_shape[1] = "<< test_shape[1];
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


