#include "saber/lite/funcs/saber_permute.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
void permute_basic(const int count, const Dtype* din, const int* permute_order, \
        const int* old_steps, const int* new_steps, const int num_axes, Dtype* dout) {
    for (int i = 0; i < count; ++i) {
        int old_idx = 0;
        int idx = i;
        for (int j = 0; j < num_axes; ++j) {
            int order = permute_order[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }
        dout[i] = din[old_idx];
    }
}

template <typename Dtype>
void transpose_mat(const Dtype* din, Dtype* dout, \
    const int num, const int width, const int height);
void transpose_mat(const float* din, float* dout, \
    const int num, const int width, const int height) {
	int nw = width >> 2;
	int nh = height >> 2;
	int size_in = width * height;

    for (int i = 0; i < num; ++i) {
        float* ptr_out = dout + i * size_in;
        const float* ptr_in = din + i * size_in;
#pragma omp parallel for
        for (int h = 0; h < nh; h++) {
            const float* ptr_din_row = ptr_in + h * 4 * width;
            for (int w = 0; w < nw; w++) {
                float* data_out_ptr = ptr_out + w * 4 * height + h * 4;
                const float* din0 = ptr_din_row;
                const float* din1 = din0 + width;
                const float* din2 = din1 + width;
                const float* din3 = din2 + width;

                float* dout0 = data_out_ptr;
                float* dout1 = dout0 + height;
                float* dout2 = dout1 + height;
                float* dout3 = dout2 + height;

#ifdef __aarch64__
                float32x4_t vr0 = vld1q_f32(din0);
                    float32x4_t vr1 = vld1q_f32(din1);
                    float32x4_t vr2 = vld1q_f32(din2);
                    float32x4_t vr3 = vld1q_f32(din3);
                    vtrnq_f32(vr0, vr1);
                    vtrnq_f32(vr2, vr3);
                    vswp_f32(d1, d4);
                    vswp_f32(d3, d6);
                    vst1q_f32(dout0, vr0);
                    vst1q_f32(dout1, vr1);
                    vst1q_f32(dout2, vr2);
                    vst1q_f32(dout3, vr3);
#else
                asm(
                "vld1.32 {d0, d1}, [%[in0]]    \n"
                        "vld1.32 {d2, d3}, [%[in1]]    \n"
                        "vld1.32 {d4, d5}, [%[in2]]    \n"
                        "vld1.32 {d6, d7}, [%[in3]]    \n"
                        "vtrn.32 q0, q1                \n"
                        "vtrn.32 q2, q3                \n"
                        "vswp d1, d4                   \n"
                        "vswp d3, d6                   \n"
                        "vst1.32 {d0, d1}, [%[out0]]   \n"
                        "vst1.32 {d2, d3}, [%[out1]]   \n"
                        "vst1.32 {d4, d5}, [%[out2]]   \n"
                        "vst1.32 {d6, d7}, [%[out3]]   \n"
                :
                : [out0] "r" (dout0), [out1] "r" (dout1), [out2] "r" (dout2), \
                        [out3] "r" (dout3), [in0] "r" (din0), [in1] "r" (din1), \
                         [in2] "r" (din2), [in3] "r" (din3)
                : "q0", "q1", "q2", "q3"
                );
#endif
                ptr_din_row += 4;
            }
        }
        //remian
        for (int h = 0; h < height; h++){
            for (int w = nw * 4; w < width; w++){
                const float* data_in_ptr = ptr_in + h * width + w;
                float* data_out_ptr = ptr_out + w * height + h;
                *data_out_ptr = *data_in_ptr;
            }
        }
        for (int w = 0; w < width; w++){
            for (int h = nh * 4; h < height; h++){
                const float* data_in_ptr = ptr_in + h * width + w;
                float* data_out_ptr = ptr_out + w * height + h;
                *data_out_ptr = *data_in_ptr;
            }
        }
    }

}

SaberPermute::SaberPermute() {
    _need_permute = false;
    _transpose = false;
}

//SaberPermute::SaberPermute(std::vector<int> orders) {
//    _order_dims = orders;
//}
//
//SaberStatus SaberPermute::load_param(std::vector<int> orders) {
//    _order_dims = orders;
//    return SaberSuccess;
//}

SaberPermute::SaberPermute(const ParamBase *param) {
    _param = (const PermuteParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberPermute::load_param(const ParamBase *param) {
    _param = (const PermuteParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPermute::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load permute param first\n");
        return SaberNotInitialized;
    }

    for (int i = 0; i < inputs.size(); ++i) {

        Shape output_shape = inputs[i]->valid_shape();

        LCHECK_EQ(inputs[i]->valid_shape().size(), _param->_order.size(), "permute order param is not valid");

        //for example: (n, h, w, c)->(n, c, h, w)  by order(0, 3, 1, 2)
        for (int j = 0; j < _param->_order.size(); j++) {
            output_shape[j] = inputs[i]->valid_shape()[_param->_order[j]];
        }
        outputs[i]->set_shape(output_shape);
    }
    return SaberSuccess;

}

//template <typename Dtype>
SaberStatus SaberPermute::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load permute param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;
    _num_axes = inputs[0]->dims();
    _count = outputs[0]->valid_size();

    LCHECK_EQ(inputs[0]->dims(), _param->_order.size(), "permute order size is not match to input dims");
    // set _need_permute
    _need_permute = false;
    for (int i = 0; i < _num_axes; ++i) {
        if (_param->_order[i] != i) {
            _need_permute = true;
            break;
        }
    }
    if (!_need_permute) {
        return SaberSuccess;
    }

    //! for basic permute
    std::vector<int> axis_diff;
    int j = 0;
    for (int i = 0; i < _num_axes; ++i) {
        if (_param->_order[j] != i) {
            axis_diff.push_back(j);
            //LOG(INFO) << "diff axis: " << _order_dims[j];
        } else {
            j++;
        }
    }
    if (axis_diff.size() == 1) {
        _transpose = true;
        _trans_num = inputs[0]->count_valid(0, std::max(axis_diff[0] - 1, 0));
        _trans_w = inputs[0]->count_valid(axis_diff[0] + 1, _num_axes);
        _trans_h = inputs[0]->valid_shape()[axis_diff[0]];
        printf("permute: transpose=true, num= %d, h=%d, w=%d\n", _trans_num , _trans_h, _trans_w);
    } else {
        _transpose = false;
        _new_steps = outputs[0]->get_stride();
        _old_steps = inputs[0]->get_stride();
        printf("permute: transpose=false\n");
    }

    this->_flag_init = true;

    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberPermute::dispatch(\
    const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {

    if (!this->_flag_init) {
        printf("init permute first\n");
        return SaberNotInitialized;
    }

    //! only copy the data
    if (!_need_permute) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();
    //! transpose the data
    if (_transpose) {
        transpose_mat(din, dout, _trans_num, _trans_w, _trans_h);
    } else {
        permute_basic(_count, din, _param->_order.data(), \
        _old_steps.data(), _new_steps.data(), _num_axes, dout);
    }

    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE