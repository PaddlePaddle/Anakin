#include "saber/lite/funcs/saber_softmax.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

#include <cmath>
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"

namespace anakin{

namespace saber{

namespace lite{

void softmax_basic(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size) {

#pragma omp parallel for
    for (int i = 0; i < compute_size; ++i) {
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }
}

//! for inner size == 1
void softmax_inner1(const float* din, float* dout, \
    const int outer_size, const int axis_size) {
#pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        const float* din_ptr = din + i * axis_size;
        float* dout_ptr = dout + i * axis_size;

        const float* din_max_ptr = din_ptr;
        int nn = axis_size >> 2;

        //! get max
        float32x4_t vmax = vld1q_f32(din_max_ptr);
        din_max_ptr += 4;
        int j = 1;
        for (; j < nn; ++j) {
            vmax = vmaxq_f32(vmax, vld1q_f32(din_max_ptr));
            din_max_ptr += 4;
        }
        float32x2_t vhmax = vmax_f32(vget_high_f32(vmax), vget_low_f32(vmax));
        float max_data = std::max(vget_lane_f32(vhmax, 0), vget_lane_f32(vhmax, 1));
        for (j = 4 * j; j < axis_size; ++j) {
            max_data = std::max(max_data, din_max_ptr[0]);
            din_max_ptr++;
        }
        //printf("max data: %.2f\n", max_data);

        //! sub, exp and sum
        const float* din_sum_ptr = din_ptr;
        float* dout_sum_ptr = dout_ptr;
        vmax = vdupq_n_f32(max_data);
        float32x4_t vsub_exp = exp_ps(vsubq_f32(vld1q_f32(din_sum_ptr), vmax));
        float32x4_t vsum = vsub_exp;
        vst1q_f32(dout_sum_ptr, vsub_exp);
        din_sum_ptr += 4;
        dout_sum_ptr += 4;

        j = 1;
        for (; j < nn; ++j) {
            vsub_exp = exp_ps(vsubq_f32(vld1q_f32(din_sum_ptr), vmax));
            vst1q_f32(dout_sum_ptr, vsub_exp);
            vsum = vaddq_f32(vsum, vsub_exp);
            din_sum_ptr += 4;
            dout_sum_ptr += 4;
        }
        float32x2_t vhsum = vadd_f32(vget_high_f32(vsum), vget_low_f32(vsum));
        float sum_data = vget_lane_f32(vhsum, 0) + vget_lane_f32(vhsum, 1);

        for (j = 4 * j; j < axis_size; ++j) {
            dout_sum_ptr[0] = expf(din_sum_ptr[0] - max_data);
            sum_data += dout_sum_ptr[0];
            din_sum_ptr++;
            dout_sum_ptr++;
        }
        //printf("sum data: %.2f\n", sum_data);

        float sum_inv = 1.f / sum_data;
        float* dout_res_ptr = dout_ptr;
        float32x4_t vinv = vdupq_n_f32(sum_inv);
        //! get softmax result
        j = 0;
        for (; j < nn; ++j) {
            float32x4_t vout = vld1q_f32(dout_res_ptr);
            float32x4_t vres= vmulq_f32(vout, vinv);
            vst1q_f32(dout_res_ptr, vres);
            dout_res_ptr += 4;
        }
        for (j = nn * 4; j < axis_size; ++j) {
            dout_ptr[j] *= sum_inv;
        }
    }
}

//! for inner size == 1 aixs_size < 4
void softmax_inner1_s(const float* din, float* dout, \
    const int outer_size, const int axis_size) {
#pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        const float* din_ptr = din + i * axis_size;
        float* dout_ptr = dout + i * axis_size;
        //! get max
        float max_data = din_ptr[0];
        for (int j =1; j < axis_size; ++j) {
            max_data = std::max(max_data, din_ptr[j]);
        }
        //printf("max data: %.2f\n", max_data);

        //! sub, exp and sum
        float sum_data = 0.f;
        for (int j = 0; j < axis_size; ++j) {
            dout_ptr[j] = expf(din_ptr[j] - max_data);
            sum_data += dout_ptr[j];
        }
        //printf("sum data: %.2f\n", sum_data);

        float sum_inv = 1.f / sum_data;
        for (int j = 0; j < axis_size; ++j) {
            dout_ptr[j] *= sum_inv;
        }
    }
}

SaberSoftmax::SaberSoftmax(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const SoftmaxParam*)param;
    this->_flag_param = true;
}

SaberSoftmax::~SaberSoftmax() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberSoftmax::load_param(const ParamBase *param) {
    _param = (const SoftmaxParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberSoftmax::load_param(std::istream &stream, const float *weights) {
    int axis;
    stream >> axis;
    _param = new SoftmaxParam(axis);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberSoftmax::load_param(FILE *fp, const float* weights) {
    int axis;
    fscanf(fp, "%d\n", &axis);
    _param = new SoftmaxParam(axis);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
SaberStatus SaberSoftmax::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load softmax param first\n");
        return SaberNotInitialized;
    }
    return outputs[0]->set_shape(inputs[0]->valid_shape());
}

SaberStatus SaberSoftmax::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load softmax param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    _outer_num = inputs[0]->count_valid(0, _param->_axis);
    _inner_num = inputs[0]->count_valid(_param->_axis + 1, inputs[0]->dims());
    _axis_size = shape_in[_param->_axis];

    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberSoftmax::dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {

    if (!this->_flag_init) {
        printf("init softmax first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    float* dout = outputs[0]->mutable_data();
    const float* din = (float*)inputs[0]->data();

    if (this->_inner_num == 1) {
        if(_axis_size >= 4){
            softmax_inner1(din, dout, _outer_num, _axis_size);
        }else{
            softmax_inner1_s(din, dout, _outer_num, _axis_size);
        }
    } else {
        int compute_size = inputs[0]->valid_size() / _axis_size;
        softmax_basic(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("softmax time: %f\n", ts);
    OpTimer::add_timer("softmax", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberSoftmax);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


