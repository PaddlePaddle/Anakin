#include "saber/core/context.h"
#include "saber/funcs/ps_roi_pooling.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <float.h>
#include <cmath>
using namespace anakin::saber;

template <typename Dtype, typename TargetType_D, typename TargetType_H>
void ps_roi_pool_cpu(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output,\
                PsRoiPoolParam<TargetType_D>& param){
    int in_n = input[0]->num();
    int in_c = input[0]->channel();
    int in_h = input[0]->height();
    int in_w = input[0]->width();
    int o_n = output[0]->num();
    int o_h = output[0]->height();
    int o_w = output[0]->width();
    int o_c = output[0]->channel();
    int pooled_h = param.pooled_height;
    int pooled_w = param.pooled_width;
    int crop_width = param.crop_width / param.pooled_width;
    int crop_height = param.crop_height / param.pooled_height;
    int num_rois = o_n;
    int im_h = in_h;
    int im_w = in_w;
    float extra_value = 0;
    int method = 0;
    int global_pooling = true;
    //float spatial_scale = param.spatial_scale;
    const Dtype* in_data = (const Dtype*)input[0]->data();
    const Dtype* rois = (const Dtype*)input[1]->data();
    Dtype* out_data = (Dtype*)output[0]->mutable_data();
    Tensor<TargetType_H> inter;
    inter.re_alloc(Shape({pooled_w*pooled_h*o_c, o_n, crop_height, crop_width}));
    Dtype* inter_data = (Dtype*)inter.mutable_data();
    int count = output[0]->valid_size();
    int inter_count = inter.valid_size();

    for (int index = 0; index < inter_count; ++index){
        int temp_ind = index;
        int cur_w = temp_ind % crop_width;
        temp_ind /= crop_width;
        int cur_h = temp_ind % crop_height;
        temp_ind /= crop_height;
        int cur_n = temp_ind % num_rois;
        int cur_c = temp_ind / num_rois;

        const Dtype* rois_data = rois + cur_n * 4;
        
        float y1 = rois_data[0] * (im_h - 1);
        float x1 = rois_data[1] * (im_w - 1);
        float y2 = rois_data[2] * (im_h - 1);
        float x2 = rois_data[3] * (im_w - 1);

        float height_scale = crop_height > 1 ? (y2 - y1) / (crop_height - 1) : 0;
        float width_scale = crop_width > 1 ? (x2 - x1) / (crop_width - 1) : 0;

        float in_y = crop_height > 1 ? y1 + cur_h * height_scale : (y1 + y2) / 2;

        if (in_y < 0 || in_y > im_h - 1){
            out_data[index] = extra_value;
            continue;
        }

        float in_x = crop_width > 1 ? x1 + cur_w * width_scale : (x1 + x2) / 2;
        if (in_x < 0 || in_x > im_w - 1){
            out_data[index] = extra_value;
            continue;
        }

        const Dtype* im_data = in_data + cur_c * im_h * im_w;

        //resize method 0 means bilinear
        if (method == 0){
            int top_y = floor(in_y);
            int bot_y = ceil(in_y);
            float y_lerp = in_y - top_y;

            int left_x = floor(in_x);
            int right_x = ceil(in_x);
            float x_lerp = in_x - left_x;

            Dtype top_left = im_data[top_y*im_w + left_x];
            Dtype top_right = im_data[top_y*im_w + right_x];
            Dtype bot_left = im_data[bot_y*im_w + left_x];
            Dtype bot_right = im_data[bot_y*im_w + right_x];
            float top = top_left + (top_right - top_left) * y_lerp;
            float bot = bot_left + (bot_right - bot_left) * y_lerp;
            inter_data[index] = top + (bot - top) * x_lerp; 
        }
    }
    int channel = o_c;
    int pooled_size = pooled_w * pooled_h;
    int crop_size = crop_height * crop_width;
    for (int index = 0; index < count; ++index){
        int cur_n = index / channel;
        int cur_c = index % channel;
        int crop_size = crop_height * crop_width;
        Dtype sum = 0;
        for (int i = 0; i < crop_size; ++i){
            Dtype tmp_sum = 0;
            for (int j = 0; j < pooled_size; ++j){
                tmp_sum += inter_data[(j * num_rois + cur_n) * crop_size + i];
            }
            sum += tmp_sum / pooled_size;
        }
        out_data[index] = sum / crop_size;
    }
    
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_ps_roi_pool(){
    typedef typename DataTrait<TargetType_D, Dtype>::Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, Dtype, PsRoiPool, PsRoiPoolParam> testbase(2, 1);
    float spatial_scale = 2.0f;
    for (auto num_in :{1, 2}){
        for (auto c_in:{4, 8}){
            for (auto h_in:{6}){
                for (auto w_in:{6}){
                    for (auto roi_num:{1, 2}){
                        for (auto pool_h:{2}){
                            for (auto pool_w:{2}){
                                for (auto ch : {2, 4}){
                                    for (auto cw : {2, 4}){
                                Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
                                Shape roi_shape({roi_num, 4, 1, 1}, Layout_NCHW);
                                Tensor<TargetType_H> th_in, th_roi;
                                Tensor<TargetType_D> td_in, td_roi;
                                th_in.re_alloc(in_shape, Dtype);
                                th_roi.re_alloc(roi_shape, Dtype);
                                td_in.re_alloc(in_shape, Dtype);
                                td_roi.re_alloc(roi_shape, Dtype);
                                // prepare host data
                                fill_tensor_rand(th_in, 0.0, 1.0);
                                // prepare roi data
                                dtype* roi_data = (dtype*)th_roi.mutable_data();
                                srand(time(0));
                                for (int i = 0; i < roi_num; ++i){
                                    //roi_data[i * 5] = rand() % num_in;
                                    roi_data[i * 4 + 0] = 0.5;
                                    roi_data[i * 4 + 1] = 0.5;
                                    roi_data[i * 4 + 2] = 1;
                                    roi_data[i * 4 + 3] = 1;
                                }
                                td_in.copy_from(th_in);
                                td_roi.copy_from(th_roi);
                                std::vector<Tensor<TargetType_D>*> input;
                                input.push_back(&td_in);
                                input.push_back(&td_roi);
                                LOG(ERROR) << num_in <<"," << c_in << ","<< h_in << ","<< w_in << ","<<
                                 roi_num << ","<< pool_h << ","<< pool_w;
                                testbase.add_custom_input(input);
                                PsRoiPoolParam<TargetType_D> param(pool_h, pool_w, ch, cw);
                                testbase.set_param(param);
                                testbase.run_test(ps_roi_pool_cpu<dtype, TargetType_D, TargetType_H>);
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

TEST(TestSaberFunc, test_func_roi_pooling){
//for (int i=0; i< 10000; ++i){
#ifdef USE_CUDA
    test_ps_roi_pool<AK_FLOAT, NV, NVHX86>();    
    LOG(INFO)<<"NV test end.";
#endif
#ifdef USE_X86_PLACE
    test_ps_roi_pool<AK_FLOAT, X86, X86>();    
    LOG(INFO)<<"X86 test end.";
#endif
//}


}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
