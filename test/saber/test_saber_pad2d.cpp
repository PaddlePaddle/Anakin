#include <vector>
#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test/saber/test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pad2d.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template<typename dtype, typename TargetType_D, typename TargetType_H>
void pad_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, \
    std::vector<Tensor<TargetType_H>*>& output, PadParam<TargetType_D>& param)
{
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());

    int in_n = input[0]->num();
    int in_c = input[0]->channel();
    int in_h = input[0]->height();
    int in_w = input[0]->width();
    int out_n = output[0]->num();
    int out_c = output[0]->channel();
    int out_h = output[0]->height();
    int out_w = output[0]->width();
    Shape in_stride = input[0]->get_stride();
    Shape out_stride = output[0]->get_stride();
    int in_idn = input[0]->num_index();
    int in_idc = input[0]->channel_index();
    int in_idh = input[0]->height_index();
    int in_idw = input[0]->width_index();
    int out_idn = output[0]->num_index();
    int out_idc = output[0]->channel_index();
    int out_idh = output[0]->height_index();
    int out_idw = output[0]->width_index();

    fill_tensor_const(*output[0], 0);

    int c0 = param.pad_c[0];
    int h0 = param.pad_h[0];
    int w0 = param.pad_w[0];
    int offset = c0 * out_stride[out_idc] + h0 * out_stride[out_idh] + w0 * out_stride[out_idw];
    for (int id = 0; id < input[0]->valid_size(); ++id){
        int i_n = (id / in_stride[in_idn]) % in_n;
        int i_c = (id / in_stride[in_idc]) % in_c;
        int i_h = (id / in_stride[in_idh]) % in_h;
        int i_w = (id / in_stride[in_idw]) % in_w;
        int out_id = i_n * out_stride[out_idn] + i_c * out_stride[out_idc] + \
                     i_h * out_stride[out_idh] + i_w * out_stride[out_idw];
        dst_ptr[out_id + offset] = src_ptr[id];
    }

}
template<typename dtype, typename TargetType_D, typename TargetType_H>
void pad_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, \
    std::vector<Tensor<TargetType_H>*>& output, Pad2DParam<TargetType_D>& param){
    const dtype* din = static_cast<dtype*>(input[0]->data());
    dtype* dout = static_cast<dtype*>(output[0]->mutable_data());
    int n = output[0]->num();
    int c = output[0]->channel();
    int h = output[0]->height();
    int w = output[0]->width();
    int pad_top = param._pad_h[0];
    int pad_bottom = param._pad_h[1];
    int pad_left = param._pad_w[0];
    int pad_right = param._pad_w[1];
    PadMode pad_mode = param._mode;
    float pad_value = param._pad_value;

    int in_w = w - pad_left - pad_right;
    int in_h = h - pad_bottom - pad_top;
    int spatial_size_out = w * h;
    int spatial_size_in = in_w * in_h;
#pragma omp parallel for
    for (int i = 0; i < n * c; ++i) {
        const float* din_batch = din + i * spatial_size_in;
        float* dout_batch = dout + i * spatial_size_out;
        int in_y = 0;
        int in_x = 0;
        for (int y = 0; y < h; ++y){
            for (int x = 0; x < w; ++x){
                switch (pad_mode){
                    case PAD_CONSTANT:
                        in_y = y - pad_top;
                        in_x = x - pad_left;
                        dout_batch[y * w + x] = (in_x >= 0 && in_x < in_w) &&  (in_y >= 0 && in_y < in_h) ? \
                                                    din_batch[in_y * in_w + in_x] : pad_value;
                        break;
                    case PAD_EDGE:
                        in_x = std::min(std::max(pad_left, x), in_w + pad_left - 1) - pad_left;
                        in_y = std::min(std::max(pad_top, y), in_h + pad_top - 1) - pad_top;
                        dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
                        break;
                    case PAD_REFLECT:
                        in_y = y - pad_top;
                        in_x = x - pad_left;
                        in_y = std::max(in_y, -in_y);
                        in_y = std::min(in_y, 2 * in_h - in_y - 2);
                        in_x = std::max(in_x, -in_x);
                        in_x = std::min(in_x, 2 * in_w - in_x - 2);
                        dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
                        break;
                    default:
                        LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
                }
            }
        }
    }
}

//test template for different device and dtype
template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pad(){
    typedef typename DataTrait<TargetType_D, OpDtype>::Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Pad2D, Pad2DParam> testbase;

    for (int pad_top : {0, 1}){
        for (int pad_bottom : {0, 1}){
            std::vector<int> pad_h{pad_top, pad_bottom};
            for (int pad_left : {0, 1}){
                for (int pad_right : {0, 1}){
                    std::vector<int> pad_w{pad_left, pad_right};
                    for (int pad_mode : {0, 1, 2}){
                        for (float pad_value : {0.f, 1.0f}){
                            Pad2DParam<TargetType_D> param(pad_h, pad_w, pad_value, pad_mode);
                            LOG(INFO) << "pad param: " << pad_mode<<" "<< pad_value<<" "<<pad_h[0]<<" "<< pad_h[1]<<" "<<pad_w[0]<<" "<< pad_w[1];
                            testbase.set_param(param);
                            for (int n : {1, 2}){
                                for (int c : {1, 3}){
                                    for (int h : {14, 24}){
                                        for (int w : {14, 24}){
                                            testbase.set_input_shape(Shape({n, c, h, w}));
                                            testbase.run_test(pad_cpu_func<dtype, TargetType_D, TargetType_H>);
                                        }
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_pad2d)
{
#ifdef USE_CUDA
    // test_pad<NV, NVHX86, AK_FLOAT>();
#endif

#ifdef USE_X86_PLACE
    // test_pad<X86, X86, AK_FLOAT>();
#endif
#ifdef USE_ARM_PLACE
    test_pad<ARM, ARM, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
