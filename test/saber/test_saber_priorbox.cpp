#include "saber/core/context.h"
#include "saber/funcs/priorbox.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <ctime>

using namespace anakin::saber;


//fc compute (native cpu version)
template <typename dtype,typename TargetType_D,typename TargetType_H>
void priorbox_cpu_base(const std::vector<Tensor<TargetType_H>* > &input, \
    std::vector<Tensor<TargetType_H>* > &output, PriorBoxParam<TargetType_D> &param) {

    int win1 = input[0]->width();
    int hin1 = input[0]->height();
    int wout = win1 * hin1 * param.prior_num * 4;
    Shape out_sh = output[0]->valid_shape();
    CHECK_EQ(out_sh[0], 1) << "output shape error";
    CHECK_EQ(out_sh[1], 2) << "output shape error";
    CHECK_EQ(out_sh[2], wout) << "output shape error";


    unsigned long long out_size = output[0]->valid_size();
    float* _cpu_data = static_cast<float*>(output[0]->mutable_data());

    float* min_buf = (float*)fast_malloc(sizeof(float) * 4);
    float* max_buf = (float*)fast_malloc(sizeof(float) * 4);
    float* com_buf = (float*)fast_malloc(sizeof(float) * param.aspect_ratio.size() * 4);

    const int width = input[0]->width();
    const int height = input[0]->height();
    int img_width = param.img_w;
    int img_height = param.img_h;
    if (img_width == 0 || img_height == 0) {
        img_width = input[1]->width();
        img_height = input[1]->height();
    }

    float step_w = param.step_w;
    float step_h = param.step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / width;
        step_h = static_cast<float>(img_height) / height;
    }
    float offset = param.offset;

    int channel_size = height * width * param.prior_num * 4;
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width;
            float box_height;
            for (int s = 0; s < param.min_size.size(); ++s) {
                int min_idx = 0;
                int max_idx = 0;
                int com_idx = 0;
                int min_size = param.min_size[s];
                //! first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size;
                //! xmin
                min_buf[min_idx++] = (center_x - box_width / 2.f) / img_width;
                //! ymin
                min_buf[min_idx++] = (center_y - box_height / 2.f) / img_height;
                //! xmax
                min_buf[min_idx++] = (center_x + box_width / 2.f) / img_width;
                //! ymax
                min_buf[min_idx++] = (center_y + box_height / 2.f) / img_height;

                if (param.max_size.size() > 0) {

                    int max_size = param.max_size[s];
                    //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrtf(min_size * max_size);
                    //! xmin
                    max_buf[max_idx++] = (center_x - box_width / 2.f) / img_width;
                    //! ymin
                    max_buf[max_idx++] = (center_y - box_height / 2.f) / img_height;
                    //! xmax
                    max_buf[max_idx++] = (center_x + box_width / 2.f) / img_width;
                    //! ymax
                    max_buf[max_idx++] = (center_y + box_height / 2.f) / img_height;
                }

                //! rest of priors
                for (int r = 0; r < param.aspect_ratio.size(); ++r) {
                    float ar = param.aspect_ratio[r];
                    if (fabsf(ar - 1.f) < 1e-6f) {
                        continue;
                    }
                    box_width = min_size * sqrt(ar);
                    box_height = min_size / sqrt(ar);
                    //! xmin
                    com_buf[com_idx++] = (center_x - box_width / 2.f) / img_width;
                    //! ymin
                    com_buf[com_idx++] = (center_y - box_height / 2.f) / img_height;
                    //! xmax
                    com_buf[com_idx++] = (center_x + box_width / 2.f) / img_width;
                    //! ymax
                    com_buf[com_idx++] = (center_y + box_height / 2.f) / img_height;
                }

                for (const auto &type : param.order) {
                    if (type == PRIOR_MIN) {
                        memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
                        idx += min_idx;
                    } else if (type == PRIOR_MAX) {
                        memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
                        idx += max_idx;
                    } else if (type == PRIOR_COM) {
                        memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
                        idx += com_idx;
                    }
                }
            }
        }
    }

    fast_free(min_buf);
    fast_free(max_buf);
    fast_free(com_buf);

    //! clip the prior's coordidate such that it is within [0, 1]
    if (param.is_clip) {
        for (int d = 0; d < channel_size; ++d) {
            _cpu_data[d] = std::min(std::max(_cpu_data[d], 0.f), 1.f);
        }
    }
    //! set the variance.

    float* ptr = _cpu_data + channel_size;
    int count = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int i = 0; i < param.prior_num; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ptr[count] = param.variance[j];
                    ++count;
                }
            }
        }
    }

}


TEST(TestSaberFunc, test_op_priorbox) {

    std::vector<float> min_size{60.f};
    std::vector<float> max_size;
    std::vector<float> aspect_ratio{2};
    std::vector<float> variance{0.1f, 0.1f, 0.2f, 0.2f};
    bool flip = true;
    bool clip = false;
    float step_h = 0;
    float step_w = 0;
    int img_w = 0;
    int img_h = 0;
    float offset = 0.5;
    std::vector<PriorType> order;

    order.push_back(PRIOR_MIN);
    order.push_back(PRIOR_MAX);
    order.push_back(PRIOR_COM);

    int width = 300;
    int height = 300;
    int channel = 3;
    int num = 1;
    int w_fea = 19;
    int h_fea = 19;
    int c_fea = 512;


    Shape sh_fea = Shape({num, c_fea, h_fea, w_fea}, Layout_NCHW);
    Shape sh_data = Shape({num, channel, height, width}, Layout_NCHW);

    std::vector<Shape> shape;
    shape.push_back(Shape({num, c_fea, h_fea, w_fea}, Layout_NCHW));
    shape.push_back(Shape({num, channel, height, width}, Layout_NCHW));

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, PriorBox, PriorBoxParam> testbase(2, 1);
    PriorBoxParam<NV> param(variance, flip, clip, img_w, img_h, step_w, step_h, offset, order, \
                        min_size, max_size, aspect_ratio, std::vector<float>(), std::vector<float>(), std::vector<float>());
    testbase.set_param(param);
    testbase.set_input_shape(shape);
    testbase.run_test(priorbox_cpu_base<float, NV, NVHX86>, 2.1e-5f);
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, PriorBox, PriorBoxParam> testbase_x86(2, 1);
    PriorBoxParam<X86> param_x86(variance, flip, clip, img_w, img_h, step_w, step_h, offset, order, \
                        min_size, max_size, aspect_ratio, std::vector<float>(), std::vector<float>(), std::vector<float>());
    testbase_x86.set_param(param_x86);
    testbase_x86.set_input_shape(shape);
    testbase_x86.run_test(priorbox_cpu_base<float, X86, X86>, 2.1e-5f);
#endif
}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
#ifdef USE_CUDA
    Env<NV>::env_init();
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}