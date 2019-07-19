#include <vector>
#include <limits>

#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test/saber/test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pixel_shuffle.h"

using namespace anakin::saber;

template<typename dtype, typename TargetType_D, typename TargetType_H>
void pixel_shuffle_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
    std::vector<Tensor<TargetType_H>*>& output,
    PixelShuffleParam<TargetType_D>& param)
{
    const float* src_ptr = static_cast<const float*>(input[0]->data());
    float* dst_ptr = static_cast<float*>(output[0]->mutable_data());
        
    int out_size = output[0]->valid_size();
    Shape in_sh = input[0]->valid_shape();

    int num_axes = input[0]->valid_shape().size() + 2;
    int rw = param.rw;
    int rh = param.rh;
    int new_c = in_sh.channel()/(rw*rh);
    std::vector<int> order;
    Shape in_new_sh;
    Shape out_new_sh;
    Shape out_sh;

    in_new_sh.push_back(in_sh.num());
    out_new_sh.push_back(in_sh.num());
    if (param.channel_first){
        in_new_sh.push_back(new_c);
        in_new_sh.push_back(param.rh);
        in_new_sh.push_back(param.rw);
        in_new_sh.push_back(in_sh.height());
        in_new_sh.push_back(in_sh.width());
        order = std::vector<int>({0, 1, 4, 2, 5, 3});
        out_new_sh.push_back(new_c);
        out_new_sh.push_back(in_sh.height());
        out_new_sh.push_back(param.rh);
        out_new_sh.push_back(in_sh.width());
        out_new_sh.push_back(param.rw);
        out_sh = Shape({in_sh.num(), new_c, 
            param.rh * in_sh.height(), param.rw * in_sh.width()});

      } else {
        in_new_sh.push_back(in_sh.height());
        in_new_sh.push_back(in_sh.width());
        in_new_sh.push_back(param.rh);
        in_new_sh.push_back(param.rw);
        in_new_sh.push_back(new_c);
        order = std::vector<int>({0, 1, 3, 2, 4, 5}); 
        out_new_sh.push_back(in_sh.height());
        out_new_sh.push_back(param.rh);
        out_new_sh.push_back(in_sh.width());
        out_new_sh.push_back(param.rw); 
        out_new_sh.push_back(new_c);
        out_sh = Shape({in_sh.num(), 
            param.rh * in_sh.height(), param.rw * in_sh.width(),  new_c});

    }
    Shape out_step = out_new_sh.get_stride();
    Shape in_step = in_new_sh.get_stride();

    if (input[0]->is_continue_mem() && output[0]->is_continue_mem()){
            for (int j=0; j<out_size; ++j){
                int in_idx = 0;
                int id = j;
                for (int i = 0; i < num_axes; ++i) {
                    int ord = order[i];
                    int new_step = out_step[i];
                    int old_step = in_step[ord];
                    int offset = (id / new_step) * old_step;
                    in_idx += offset;
                    id %= new_step;
                }
                dst_ptr[j] = src_ptr[in_idx];
            }
        } else {
            for (int j=0; j<out_size; ++j){
                int in_idx = 0;
                int out_idx  = 0;
                int new_valid_stride = 1;
                for (int i = num_axes - 1; i >= 0; --i) {
                    int ord = order[i];
                    int new_step = out_step[i];
                    int old_step = in_step[ord];
                    int id = (j / new_valid_stride) % out_new_sh[i];
                    in_idx += id * old_step;
                    out_idx += id * new_step;
                    new_valid_stride *= out_new_sh[i];
                }
                dst_ptr[out_idx] = src_ptr[in_idx];
            }
        }

        output[0]->set_shape(out_sh);

}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pixel_shuffle(){
    typedef typename DataTrait<TargetType_H, OpDtype> :: Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, PixelShuffle, PixelShuffleParam> testbase;
    for (int rw : {2, 3, 4}){
        for (int rh : {2, 3, 4}){
            PixelShuffleParam<TargetType_D> param(rh, rw);
            for (int n : {1, 3}){
                for (int c : {144, 288}){
                    for (int h : {8, 32}){
                        for (int w: {8, 32}){
                            testbase.set_param(param);
                            testbase.set_input_shape(Shape({n, c, h, w}));
                            testbase.run_test(pixel_shuffle_cpu_func<dtype, TargetType_D, TargetType_H>);
                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_permute)
{
#ifdef USE_CUDA
    test_pixel_shuffle<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_pixel_shuffle<X86, X86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
