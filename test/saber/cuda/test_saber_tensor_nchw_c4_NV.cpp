#include "test_saber_tensor_NV.h"
#include "tensor_op.h"
#include <vector>
using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef TargetWrapper<NV> NV_API;
typedef Tensor<X86, AK_INT8, NCHW_C4> TensorHf4;
typedef Tensor<NV, AK_INT8, NCHW_C4> TensorDf4;
typedef TensorHf4::Dtype dtype;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor& t0) {

    LOG(INFO) << name << " valid shape is ["
              << t0.valid_shape()[0] << ", "
              << t0.valid_shape()[1] << ", "
              << t0.valid_shape()[2] << ", "
              << t0.valid_shape()[3] << ", "
              << t0.valid_shape()[4] << "].";

    LOG(INFO) << name << " real shape is ["
              << t0.shape()[0] << ", "
              << t0.shape()[1] << ", "
              << t0.shape()[2] << ", "
              << t0.shape()[3] << ", "
              << t0.shape()[4] << "].";

    LOG(INFO) << name << " offset is ["
              << t0.offset()[0] << ", "
              << t0.offset()[1] << ", "
              << t0.offset()[2] << ", "
              << t0.offset()[3] << ", "
              << t0.offset()[4] << "].";
}

TEST(TestSaberTensorNV, test_tensor_constructor) {

    //! test empty constructor
    TensorHf4 thost0;
    TensorDf4 tdev0;
    Shape nchw_c4_shape(1, 4, 4, 4, 4);
    thost0.re_alloc(nchw_c4_shape);
    tdev0.re_alloc(nchw_c4_shape);
    print_tensor_device(tdev0);
    cudaDeviceSynchronize();

    fill_tensor_host_rand(thost0);
    print_tensor_host(thost0);

    tdev0.copy_from(thost0);
    print_tensor_device(tdev0);
    cudaDeviceSynchronize();
    LOG(INFO) << "thost0 channel " << thost0.channel()
              << ", thost0 channel_idx " << thost0.channel_index();
    CHECK_EQ(thost0.channel(), 16);
    LOG(INFO) << "tdev0 channel " << tdev0.channel()
              << " , tdev channel_idx " << tdev0.channel_index();
    CHECK_EQ(tdev0.channel(), 16);
    CHECK_EQ(tdev0.channel_index(), 1);

    LOG(INFO) << "thost0.dims " << thost0.dims()
              << " , tdev .dims " << tdev0.dims();

}

TEST(TestSaberTensorNV, test_tensor_share) {

    //! test empty constructor
    TensorHf4 thost0;
    TensorDf4 tdev0;
    TensorHf4 tsharehost1;
    TensorDf4 tsharedev1;

    Shape nchw_c4_shape(1, 4, 4, 4, 4);

    thost0.re_alloc(nchw_c4_shape);
    tdev0.re_alloc(nchw_c4_shape);

    print_tensor_device(tdev0);
    cudaDeviceSynchronize();

    fill_tensor_host_rand(thost0);
    print_tensor_host(thost0);

    tdev0.copy_from(thost0);
    print_tensor_device(tdev0);
    cudaDeviceSynchronize();

    Shape nchw_share_shape(1, 4, 2, 2, 4);

    tsharedev1.share_sub_buffer(tdev0, nchw_share_shape, {0, 0, 0, 0, 0});
    print_tensor_shape("tsharedev1", tsharedev1);

    tsharehost1.share_sub_buffer(thost0, nchw_share_shape, {0, 0, 2, 2, 0});
    print_tensor_shape("tsharehost1", tsharehost1);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv) {
    // initial logger
    //    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

