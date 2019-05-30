
#include "saber/funcs/impl/x86/saber_scale.h"
#include <immintrin.h>
#include "saber/funcs/impl/x86/saber_avx2_expand.h"
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ScaleParam<X86>& param,
    Context<X86>& ctx) {

    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ScaleParam<X86>& param,
    Context<X86>& ctx) {
    return SaberSuccess;
}

#if defined(__AVX2__)
inline void avx2_scale_nchwc8(const float* data_in_ptr, float* data_out_ptr, int outer_id, int outer_dim,
                              int scale_dim, int inner_dimm, const float* scale_ptr, const float* bias_ptr) {
    const int round_dim = inner_dimm * 8;
    CHECK_EQ(scale_dim % 8, 0);
    const float* data_in = data_in_ptr + outer_id * scale_dim * inner_dimm;
    float* data_out = data_out_ptr + outer_id * scale_dim * inner_dimm;

    if (bias_ptr != nullptr) {
#pragma omp parallel for schedule(static)
        for (int scale_id = 0; scale_id < scale_dim; scale_id += 8) {
            const int scale_offset = scale_id * inner_dimm;
            __m256 bias = _mm256_loadu_ps(&bias_ptr[scale_id]);
            __m256 scale = _mm256_loadu_ps(&scale_ptr[scale_id]);
            const int end_iter = scale_offset + round_dim;

            for (int i = scale_offset; i < end_iter; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                x = _mm256_fmadd_ps(scale, x, bias);
                _mm256_storeu_ps(&data_out[i], x);
            }
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int scale_id = 0; scale_id < scale_dim; scale_id += 8) {
            const int scale_offset = scale_id * inner_dimm;
            __m256 scale = _mm256_loadu_ps(&scale_ptr[scale_id]);
            const int end_iter = scale_offset + round_dim;

            for (int i = scale_offset; i < end_iter; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                x = _mm256_mul_ps(scale, x);
                _mm256_storeu_ps(&data_out[i], x);
            }
        }
    }
}


inline void avx2_scale(const float* data_in_ptr, float* data_out_ptr, int outer_id, int outer_dim,
                       int scale_dim, int inner_dimm, const float* scale_ptr, const float* bias_ptr) {
    int round_dim = inner_dimm / 8 * 8;
    int remainder = inner_dimm % 8;
    const float* data_in = data_in_ptr + outer_id * scale_dim * inner_dimm;
    float* data_out = data_out_ptr + outer_id * scale_dim * inner_dimm;

    if (bias_ptr != nullptr) {
        __m256i _vec_mask = _m256_continue_mask_m256i(remainder);

        for (int scale_id = 0; scale_id < scale_dim; scale_id++) {
            const int scale_offset = scale_id * inner_dimm;
            __m256 bias = _mm256_set1_ps(bias_ptr[scale_id]);
            __m256 scale = _mm256_set1_ps(scale_ptr[scale_id]);
            const int end_iter = scale_offset + round_dim;

            for (int i = scale_offset; i < end_iter; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                x = _mm256_fmadd_ps(scale, x, bias);
                _mm256_storeu_ps(&data_out[i], x);
            }

            if (remainder > 0) {
                __m256 x = _mm256_maskload_ps(&data_in[end_iter], _vec_mask);
                x = _mm256_fmadd_ps(scale, x, bias);
                _mm256_maskstore_ps(&data_out[end_iter], _vec_mask, x);
            }
        }
    } else {
        __m256i _vec_mask = _m256_continue_mask_m256i(remainder);

        for (int scale_id = 0; scale_id < scale_dim; scale_id++) {
            const int scale_offset = scale_id * inner_dimm;
            __m256 scale = _mm256_set1_ps(scale_ptr[scale_id]);
            const int end_iter = scale_offset + round_dim;

            for (int i = scale_offset; i < end_iter; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                x = _mm256_mul_ps(scale, x);
                _mm256_storeu_ps(&data_out[i], x);
            }

            if (remainder > 0) {
                __m256 x = _mm256_maskload_ps(&data_in[end_iter], _vec_mask);
                x = _mm256_mul_ps(scale, x);
                _mm256_maskstore_ps(&data_out[end_iter], _vec_mask, x);
            }
        }
    }

}
#endif

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ScaleParam<X86>& param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_op;

    const DataType_op* in_data = (const DataType_op*)inputs[0]->data();
    DataType_op* out_data = (DataType_op*)outputs[0]->mutable_data();
    //const DataTensor_in* in_scale = (inputs.size() > 1) ? (const DataTensor_in*)inputs[1] : (const DataTensor_in*)param.scale_w;
    //const DataType_op* scale_data = (const DataType_op*)in_scale->data();
    const DataType_op* scale_data = (inputs.size() > 1) ? (const DataType_op*)inputs[1]->data() : &
                                    (param.scale_w[0]);
    const DataType_op* bias_data = param.bias_term ? &(param.scale_b[0]) : NULL;

    const int count = inputs[0]->valid_size();
    int axis = (param.num_axes == 0) ? 0 : param.axis;
    int num_axes = param.num_axes >= 0 ? param.num_axes : inputs[0]->shape().dims() - axis;
    CHECK_LE(axis + num_axes, inputs[0]->shape().dims());
    int outer_dim = inputs[0]->count_valid(0, axis);
    int inner_dim = inputs[0]->count_valid(axis + num_axes, inputs[0]->shape().dims());
    int scale_dim = inputs[0]->count_valid(axis, axis + num_axes);
    if (inputs.size() > 1) {
        scale_dim = inputs[1]->valid_size();
        inner_dim = count / scale_dim;    
    } else {
        CHECK_EQ(scale_dim, param.scale_w.size()) << "scale dim not valid";
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    // TODO !! need add other types of scale

    if (avx2_can_used()) {
#if defined(__AVX2__)
        if (inputs[0]->get_layout() == Layout_NCHW_C8R) {
            CHECK_EQ(outputs[0]->get_layout(), Layout_NCHW_C8R);

            if (scale_dim == 1) {
                for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
                    avx2_scale(in_data, out_data, outer_id, outer_dim, scale_dim, inner_dim, scale_data, bias_data);
                }
            } else {
                for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
                    avx2_scale_nchwc8(in_data, out_data, outer_id, outer_dim, scale_dim, inner_dim, scale_data,
                                      bias_data);
                }
            }
        } else {
            for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
                avx2_scale(in_data, out_data, outer_id, outer_dim, scale_dim, inner_dim, scale_data, bias_data);
            }
        }
#endif
    } else {
        for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
            for (int scale_id = 0; scale_id < scale_dim; scale_id++) {
                auto scale = scale_data[scale_id];
                auto bias = param.bias_term ? bias_data[scale_id] : 0;
                
                for (int inner_id = 0; inner_id < inner_dim; inner_id++) {
                    *out_data = (*in_data) * scale + bias;
                    in_data++;
                    out_data++;
                }
            }
        }
    }

    return SaberSuccess;
}

template class SaberScale<X86, AK_FLOAT>;

}
} // namespace anakin
