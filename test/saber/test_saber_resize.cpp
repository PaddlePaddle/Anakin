
#include "saber/core/context.h"
#include "saber/funcs/resize.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

#include <chrono>


using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void resize_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                ResizeParam<TargetType_D>& param){
    int win = input[0]->width();
    int hin = input[0]->height();
    int channels = input[0]->channel();
    int num = input[0]->num();
    int wout = output[0]->width();
    int hout = output[0]->height();
    dtype scale_w = 1 / param.width_scale;
    dtype scale_h = 1 / param.height_scale;
    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    int dst_stride_w = 1, dst_stride_h = wout, dst_stride_c = wout * hout, dst_stride_batch = wout * hout * channels;
    int src_stride_w = 1, src_stride_h = win, src_stride_c = win * hin, src_stride_batch = win * hin * channels;
    for(int n = 0; n < num; ++n){
        for(int c = 0; c < channels; ++c){
            int src_index = n * src_stride_batch + c * src_stride_c;
            for(int h = 0; h < hout; ++h){
                for(int w = 0; w < wout; ++w){
                    dtype fw = w * scale_w;
                    dtype fh = h * scale_h;
                    int w_start = (int)fw;
                    int w_end = (int)fw + 1;
                    int h_start = (int)fh;
                    int h_end = (int)fh + 1;
                    fw -= w_start;
                    fh -= h_start;
                    const dtype w00 = (1.0 - fh) * (1.0 - fw);
                    const dtype w01 = fw * (1.0 - fh);
                    const dtype w10 = fh * (1.0 - fw);
                    const dtype w11 = fw * fh;
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = w_end >= win ? 0 : src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = h_end >= hin ? 0 : src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = (w_end >= win) || (h_end >= hin) ? 0 : src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_c + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
   
}


TEST(TestSaberFunc, test_func_resize){

#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV, NVHX86, AK_FLOAT, Resize, ResizeParam> testbase;
    for (int num_in : {1, 3, 5, 8}) {
        for (int c_in : {1, 3, 5, 8}) {
            for (int h_in : {2, 3, 5, 8}) {
                for (int w_in : {2, 3, 5, 8}) {
                    for (float scale_w : {1.0f, 2.0f, 3.3f}) {
                        for (float scale_h : {1.0f, 2.0f, 4.4f}) {
                                LOG(INFO) << scale_w << "   " << scale_h;
                                ResizeParam<NV> param(scale_w, scale_h);
                                testbase.set_param(param);
                                testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                                testbase.run_test(resize_cpu<float, NV, NVHX86>);
                        }
                    }
                }
            }
        }
    }

    
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
