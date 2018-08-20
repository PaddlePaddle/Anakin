#include <vector>

#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test/saber/test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pad.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template<typename dtype,typename TargetType_D,typename TargetType_H>
void pad_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output, PadParam<TargetType_D>& param)
{
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());
    
    int in_n = input[0] -> num();
    int in_c = input[0] -> channel();
    int in_h = input[0] -> height();
    int in_w = input[0] -> width();
    int out_n = output[0] -> num();
    int out_c = output[0] -> channel();
    int out_h = output[0] -> height();
    int out_w = output[0] -> width();
    Shape in_stride = input[0] -> get_stride();
    Shape out_stride = output[0] -> get_stride();
    int in_idn = input[0] -> num_index();
    int in_idc = input[0] -> channel_index();
    int in_idh = input[0] -> height_index();
    int in_idw = input[0] -> width_index();
    int out_idn = output[0] -> num_index();
    int out_idc = output[0] -> channel_index();
    int out_idh = output[0] -> height_index();
    int out_idw = output[0] -> width_index();
    
    fill_tensor_const(*output[0], 0);
    
    int c0 = param.pad_c[0];
    int h0 = param.pad_h[0];
    int w0 = param.pad_w[0];
    int offset = c0 * out_stride[out_idc] + h0 * out_stride[out_idh] + w0 * out_stride[out_idw];
    for (int id = 0; id < input[0] -> valid_size(); ++id){
        int i_n = (id / in_stride[in_idn]) % in_n;
        int i_c = (id / in_stride[in_idc]) % in_c;
        int i_h = (id / in_stride[in_idh]) % in_h;
        int i_w = (id / in_stride[in_idw]) % in_w;
        int out_id = i_n * out_stride[out_idn] + i_c * out_stride[out_idc] + i_h * out_stride[out_idh] + i_w * out_stride[out_idw];
        dst_ptr[out_id + offset] = src_ptr[id];
    }
    
}

//test template for different device and dtype
template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pad(){
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype , Pad, PadParam> testbase;
    
    for (int pad_c0 : {0, 1, 2}){
        for (int pad_c1 : {0, 1, 2}){
            std::vector<int> pad_c{pad_c0, pad_c1};
            for (int pad_h0 : {0, 1, 2}){
                for (int pad_h1 : {0, 1, 2}){
                    std::vector<int> pad_h{pad_h0, pad_h1};
                    for (int pad_w0 : {0, 1, 2}){
                        for (int pad_w1 : {0, 1, 2}){
                            std::vector<int> pad_w{pad_w0, pad_w1};
                            PadParam<TargetType_D> param(pad_c, pad_h, pad_w);
                            LOG(INFO)<<pad_c[0]<<" "<< pad_c[1]<<" "<<pad_h[0]<<" "<< pad_h[1]<<" "<<pad_w[0]<<" "<< pad_w[1];
                            testbase.set_param(param);
                            for (int n : {1, 2}){
                                for (int c : {1, 3}){
                                    for (int h : {32, 64}){
                                        for (int w : {32, 64}){
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

TEST(TestSaberFunc, test_func_pool)
{
#ifdef USE_CUDA
    test_pad<NV, NVHX86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
