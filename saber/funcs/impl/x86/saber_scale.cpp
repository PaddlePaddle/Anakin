
#include "saber/funcs/impl/x86/saber_scale.h"
#include <immintrin.h>
#include "saber/funcs/impl/x86/saber_avx2_expand.h"
#include "saber/funcs/timer.h"
namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<X86> &param,
        Context<X86> &ctx)
{

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
        ScaleParam<X86> &param,
        Context<X86> &ctx)
{
    return SaberSuccess;
}

/*
inline avx2_scale_inner_dim_1(float* data_in_ptr,float* data_out_ptr,int batch,int length,float* scale_ptr,float* bias_ptr){
    int round_dim=length/8*8;
    int remainder=length%8;
    if(bias_ptr!= nullptr) {
        for (int batch_id = 0; batch_id < batch; batch_id++) {
            const float* data_in=data_in+batch_id*length;
            float* data_out=data_out_ptr+batch_id*length;
            for (int i = 0; i < round_dim; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                __m256 bias = _mm256_loadu_ps(&bias_ptr[i]);
                __m256 scale = _mm256_loadu_ps(&scale_ptr[i]);
                __m256 ans = _mm256_fmadd_ps(scale, x, bias);
                _mm256_storeu_ps(&data_out[i], ans);
            }
            if (remainder > 0) {
                __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                __m256 x = _mm256_maskload_ps(&data_in[round_dim], _vec_mask);
                __m256 bias = _mm256_maskload_ps(&bias_ptr[round_dim], _vec_mask);
                __m256 scale = _mm256_maskload_ps(&scale_ptr[round_dim], _vec_mask);
                __m256 ans = _mm256_fmadd_ps(scale, x, bias);
                _mm256_maskstore_ps(&data_out[round_dim], _vec_mask, ans);
            }
        }
    }else{
        for (int batch_id = 0; batch_id < batch; batch_id++) {
            const float* data_in=data_in+batch_id*length;
            float* data_out=data_out_ptr+batch_id*length;
            for (int i = 0; i < round_dim; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                __m256 scale = _mm256_loadu_ps(&scale_ptr[i]);
                __m256 ans = _mm256_mul_ps(scale, x);
                _mm256_storeu_ps(&data_out[i], ans);
            }
            if (remainder > 0) {
                __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                __m256 x = _mm256_maskload_ps(&data_in[round_dim], _vec_mask);
                __m256 bias = _mm256_maskload_ps(&bias_ptr[round_dim], _vec_mask);
                __m256 scale = _mm256_maskload_ps(&scale_ptr[round_dim], _vec_mask);
                __m256 ans = _mm256_mul_ps(scale, x);
                _mm256_maskstore_ps(&data_out[round_dim], _vec_mask, ans);
            }
        }
    }

}

inline avx2_scale_inner_dim_1(float* data_in_ptr,float* data_out_ptr,int batch,int length,float* scale_ptr,float* bias_ptr){
    int round_dim=length/8*8;
    int remainder=length%8;
    if(bias_ptr!= nullptr) {
        for (int batch_id = 0; batch_id < batch; batch_id++) {
            const float* data_in=data_in+batch_id*length;
            float* data_out=data_out_ptr+batch_id*length;
            for (int i = 0; i < round_dim; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                __m256 bias = _mm256_loadu_ps(&bias_ptr[i]);
                __m256 scale = _mm256_loadu_ps(&scale_ptr[i]);
                __m256 ans = _mm256_fmadd_ps(scale, x, bias);
                _mm256_storeu_ps(&data_out[i], ans);
            }
            if (remainder > 0) {
                __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                __m256 x = _mm256_maskload_ps(&data_in[round_dim], _vec_mask);
                __m256 bias = _mm256_maskload_ps(&bias_ptr[round_dim], _vec_mask);
                __m256 scale = _mm256_maskload_ps(&scale_ptr[round_dim], _vec_mask);
                __m256 ans = _mm256_fmadd_ps(scale, x, bias);
                _mm256_maskstore_ps(&data_out[round_dim], _vec_mask, ans);
            }
        }
    }else{
        for (int batch_id = 0; batch_id < batch; batch_id++) {
            const float* data_in=data_in+batch_id*length;
            float* data_out=data_out_ptr+batch_id*length;
            for (int i = 0; i < round_dim; i += 8) {
                __m256 x = _mm256_loadu_ps(&data_in[i]);
                __m256 scale = _mm256_loadu_ps(&scale_ptr[i]);
                __m256 ans = _mm256_mul_ps(scale, x);
                _mm256_storeu_ps(&data_out[i], ans);
            }
            if (remainder > 0) {
                __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                __m256 x = _mm256_maskload_ps(&data_in[round_dim], _vec_mask);
                __m256 bias = _mm256_maskload_ps(&bias_ptr[round_dim], _vec_mask);
                __m256 scale = _mm256_maskload_ps(&scale_ptr[round_dim], _vec_mask);
                __m256 ans = _mm256_mul_ps(scale, x);
                _mm256_maskstore_ps(&data_out[round_dim], _vec_mask, ans);
            }
        }
    }

}
*/

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<X86> &param)
{
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_op;

    const DataType_op* in_data = (const DataType_op*)inputs[0]->data();
    DataType_op* out_data = (DataType_op*)outputs[0]->mutable_data();
    const DataType_op* scale_data = (inputs.size() > 1) ? (const DataType_op*)inputs[1]->data() : &(param.scale_w[0]);
    const DataType_op* bias_data = param.bias_term ? &(param.scale_b[0]) : NULL;

    const int count = inputs[0]->valid_size();
    int axis = (param.num_axes == 0) ? 0 : param.axis;
    int num_axes = param.num_axes >=0 ? param.num_axes : inputs[0]->shape().dims() - axis;
    CHECK_LE(axis + num_axes, inputs[0]->shape().dims());
    int outer_dim = inputs[0]->count_valid(0, axis);
    int inner_dim = inputs[0]->count_valid(axis + num_axes, inputs[0]->shape().dims());
    int scale_dim = inputs[0]->count_valid(axis, axis + num_axes);
    if (inputs.size() > 1) {
        CHECK_EQ(scale_dim, inputs[1]->valid_size()) << "scale dim not valid";
    } else {
        CHECK_EQ(scale_dim, param.scale_w.size()) << "scale dim not valid";
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
//    LOG(INFO)<<"outer_dim "<<outer_dim<<",inner_dim "<<inner_dim<<",scale_dim "<<scale_dim;
//    if(inner_dim==1){
//        avx2_scale_eltwise(in_data,out_data,outer_dim,scale_dim,scale_data,bias_data);
//        return SaberSuccess;
//    }else{
//        int round_dim=inner_dim/8*8;
//        int remainder=inner_dim%8;
//        for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
//            for (int scale_id = 0; scale_id < scale_dim; scale_id++) {
//                __m256 scale = _mm256_set1_ps(scale_data[scale_id]);
//                for (int i = 0; i < round_dim; i += 8) {
//                    __m256 x = _mm256_loadu_ps(&data_in_ptr[i]);
//                    __m256 ans = _mm256_mul_ps(scale, x);
//                    _mm256_storeu_ps(&data_out_ptr[i], ans);
//                }
//                if (remainder > 0) {
//                    __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
//                    __m256 x = _mm256_maskload_ps(&data_in_ptr[round_dim], _vec_mask);
//                    __m256 bias = _mm256_maskload_ps(&bias_ptr[round_dim], _vec_mask);
//                    __m256 scale = _mm256_maskload_ps(&scale_ptr[round_dim], _vec_mask);
//                    __m256 ans = _mm256_mul_ps(scale, x);
//                    _mm256_maskstore_ps(&data_out_ptr[round_dim], _vec_mask, ans);
//                }
//
//            }
//        }
//    }
    // TODO !! need add other types of scale

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



    return SaberSuccess;
}

template class SaberScale<X86, AK_FLOAT>;

}
} // namespace anakin
