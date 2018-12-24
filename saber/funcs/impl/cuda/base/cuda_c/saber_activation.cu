#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/core/tensor_op.h"
#define BUILD_DEV __device__

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_relu_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count, Dtype neg_slop,
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                             int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride
                     + c * in_c_stride
                     + h * in_h_stride
                     + w * in_w_stride;

        int out_idx =  n * out_n_stride
                       + c * out_c_stride
                       + h * out_h_stride
                       + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        out_data[out_idx] = in_var > Dtype(0) ? in_var : in_var * neg_slop;
    }
}

template<typename Dtype>
__global__ void ker_sigmoid_fwd(Dtype * out_data,
                                const Dtype* in_data, const int count,
                                int in_n, int in_c, int in_h, int in_w,
                                int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                                int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {

    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        Dtype in_var = in_data[in_idx];

        out_data[out_idx] = Dtype( Dtype(1) / (Dtype(1)+ exp(-in_var)));

    }
}

template<typename Dtype>
__global__ void ker_tanh_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count,
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                             int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {

    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        //(expf(in_var) - expf(-in_var)) / (expf(in_var) + expf(-in_var));exp
        out_data[out_idx] = Dtype(1) - (Dtype(2) / (Dtype(1) + exp(in_var * 2))); 

    }
}

template<typename Dtype>
__global__ void ker_stanh_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count, const Dtype slope, const Dtype coef, 
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                             int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {

    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;


        Dtype in_var = in_data[in_idx];
        Dtype var = in_var * slope;
        //output_data[j] = param.coef * tanh(param.negative_slope * input_data[j]);
        out_data[out_idx] = Dtype( coef * (Dtype(1) - (Dtype(2) / (Dtype(1) + exp(var * 2)))));
    }
}

template<typename Dtype>
__global__ void ker_clipped_relu_fwd(Dtype * out_data,
                                     const Dtype* in_data, const int count, Dtype clipped_threadhold,
                                     int in_n, int in_c, int in_h, int in_w,
                                     int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                                     int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        in_var = in_var > 0 ? in_var : 0;
        out_data[out_idx] = in_var < clipped_threadhold? in_var : clipped_threadhold;
    }
}
template<typename Dtype>
__global__ void ker_elu_fwd(Dtype * out_data,
                            const Dtype* in_data, const int count, Dtype coef,
                            int in_n, int in_c, int in_h, int in_w,
                            int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                            int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count){
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        out_data[out_idx] = in_var > 0 ? in_var : coef * (exp(in_var)-1);
    }
}

template<typename Dtype>
__global__ void ker_prelu_fwd(Dtype * out_data,
                              const Dtype* in_data, const int count,
                              const Dtype* slope, bool is_channel_shared,
                              int in_n, int in_c, int in_h, int in_w,
                              int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                              int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count){
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        if (is_channel_shared) {
            out_data[out_idx] = in_var > 0 ? in_var : slope[0] * in_var;
        } else {
            out_data[out_idx] = in_var > 0 ? in_var : slope[c] * in_var;
        }
    }
}

template <>
SaberStatus SaberActivation<NV, AK_FLOAT>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberActivation<NV, AK_FLOAT>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberActivation<NV, AK_FLOAT>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param) {

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    Shape stride_in = inputs[0]->get_stride();
    Shape stride_out = outputs[0]->get_stride();

    const float *in_data = (const float*)inputs[0]->data();
    float *out_data = (float*)outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    float negative_slope = param.negative_slope;
    float coef = param.coef;
    switch (param.active) {
        //x > 0 ? x : 0
        case Active_relu:

            ker_relu_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, negative_slope,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;

        // sigmoid: 1/(exp(-x) + 1)
        case Active_sigmoid:

            ker_sigmoid_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;

        // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        case Active_tanh:
        
            ker_tanh_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        
        // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
        case Active_stanh:

            ker_stanh_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, negative_slope, coef, 
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;

        // x > 0 ? x : 0;
        // x < threshold ? x : threshold
        case Active_clipped_relu:

            ker_clipped_relu_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, coef,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;

        //elu:  x > 0 ? x : coef * (exp(x) - 1)
        case Active_elu:

            ker_elu_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, coef,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        //prelu: x > 0 ? x : slope[c] * x
        case Active_prelu:
            auto prelu_param  = param.prelu_param;
            const float* slope_ptr = (const float*)prelu_param.slope->data();
            bool shared = prelu_param.channel_shared;
            ker_prelu_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                            out_data, in_data, count, 
                            slope_ptr, shared,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
    }
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

// =================================int8 ==================
class ReluDev{
public:
    static __device__ float run(float in, float negative_slope, float placeholder) {
        return (in > 0.f) ? in : in * negative_slope;
    }
};
class SigmoidDev{
public:
    static __device__ float run(float in, float placeholder1, float placeholder2) {
        return float( float(1) / (float(1)+ exp(-in)));
    }
};

template <typename Op>
__global__
void ker_act_fwd_fp32_to_int8(char* out_data, const float* in_data,
        int in_num, int in_channel_4, int in_height, int in_width,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
        const float negtive_slope, const float coef, float scale, int count) {

    int load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int write_w = (gid) % in_width;
    int write_h = (gid / (out_h_stride)) % in_height;
    int write_c = (gid / (out_c_stride)) % in_channel_4;
    int write_n = (gid / (out_n_stride)) % in_num;

    int in_offset = write_n * in_n_stride
                    + write_c * in_c_stride * 4
                    + write_h * in_h_stride
                    + write_w * in_w_stride;

    int out_offset = write_n * out_n_stride
                     + write_c * out_c_stride
                     + write_h * out_h_stride
                     + write_w;

    if (gid < count) {
        char4 write;
        float temp;
        temp = in_data[in_offset] * scale;
        temp = Op::run(temp, negtive_slope, coef);
        load0 = __float2int_rn(temp);
        write.x = static_cast<char>(load0);

        in_offset += in_c_stride;
        temp = in_data[in_offset] * scale;
        temp = Op::run(temp, negtive_slope, coef);
        load1 = __float2int_rn(temp);
        write.y = static_cast<char>(load1);

        in_offset += in_c_stride;
        temp = in_data[in_offset] * scale;
        temp = Op::run(temp, negtive_slope, coef);
        load2 = __float2int_rn(temp);
        write.z = static_cast<char>(load2);

        in_offset += in_c_stride;
        temp = in_data[in_offset] * scale;
        temp = Op::run(temp, negtive_slope, coef);
        load3 = __float2int_rn(temp);
        write.w = static_cast<char>(load3);

        ((char4*)out_data)[out_offset] = write;
    }
}

template <typename Op>
__global__
void ker_act_fwd_int8_to_fp32(float* out_data, const char* in_data,
        int in_num, int in_channel_4, int in_height, int in_width,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
        const float negtive_slope, const float coef, const float scale, int count) {

    float load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % in_width;
    int read_h = (gid / (in_h_stride)) % in_height;
    int read_c = (gid / (in_c_stride)) % in_channel_4;
    int read_n = (gid / (in_n_stride)) % in_num;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {
        char4 readin = ((const char4*)in_data)[in_offset];
        load0 = static_cast<float>(readin.x) * scale;
        load1 = static_cast<float>(readin.y) * scale;
        load2 = static_cast<float>(readin.z) * scale;
        load3 = static_cast<float>(readin.w) * scale;
        load0 = Op::run(load0, negtive_slope, coef);
        load1 = Op::run(load1, negtive_slope, coef);
        load2 = Op::run(load2, negtive_slope, coef);
        load3 = Op::run(load3, negtive_slope, coef);
        out_data[out_offset] = load0; out_offset += out_c_stride;
        out_data[out_offset] = load1; out_offset += out_c_stride;
        out_data[out_offset] = load2; out_offset += out_c_stride;
        out_data[out_offset] = load3;
    }
}

__global__ void ker_sigmoid_fwd_int8(char * out_data,
                                const char* in_data, const int count,
                                int in_n, int in_c, int in_h, int in_w,
                                int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                                int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                                float in_scale = 1.f, float out_scale = 1.f) {

    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx =   n * in_n_stride
                       + c * in_c_stride
                       + h * in_h_stride
                       + w * in_w_stride;

        int out_idx =   n * out_n_stride
                        + c * out_c_stride
                        + h * out_h_stride
                        + w * out_w_stride;

        char in_var = in_data[in_idx];
        float in = static_cast<float>(in_var) * in_scale;
        in = float( float(1) / (float(1)+ exp(-in)));
        in /= out_scale;
        out_data[out_idx] = static_cast<char>(in);
    }
}

template <>
SaberStatus SaberActivation<NV, AK_INT8>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberActivation<NV, AK_INT8>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberActivation<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ActivationParam<NV>& param) {

    cudaStream_t cuda_stream = _ctx->get_compute_stream();
    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    Shape stride_in = inputs[0]->get_stride();
    Shape stride_out = outputs[0]->get_stride();

    const void *in_data = inputs[0]->data();
    void *out_data = outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    int in_c_4 = inputs[0]->channel() / 4;
    int out_c_4 = outputs[0]->channel() / 4;
    float in_scale = inputs[0]->get_scale()[0];
    float out_scale = outputs[0]->get_scale()[0];

    float negative_slope = param.negative_slope;
    float coef = param.coef;

    if (inputs[0]->get_dtype() == AK_FLOAT && outputs[0]->get_dtype() == AK_INT8) {
        switch (param.active) {
        //x > 0 ? x : 0
        case Active_relu:
            ker_act_fwd_fp32_to_int8<ReluDev>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
                    (char*)out_data, (const float*)in_data,
                    outputs[0]->num(), out_c_4, outputs[0]->height(), outputs[0]->width(),
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_c_4 * outputs[0]->height() * outputs[0]->width(),
                    outputs[0]->height() * outputs[0]->width(), outputs[0]->width(), 1,
                    negative_slope, coef, (1.f / out_scale), count);
            break;
        case Active_sigmoid:
            ker_act_fwd_fp32_to_int8<SigmoidDev>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
                    (char*)out_data, (const float*)in_data,
                    outputs[0]->num(), out_c_4, outputs[0]->height(), outputs[0]->width(),
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_c_4 * outputs[0]->height() * outputs[0]->width(),
                    outputs[0]->height() * outputs[0]->width(), outputs[0]->width(), 1,
                    negative_slope, coef, (1.f / out_scale), count);
            break;
        default:
            LOG(FATAL) << "Not implement this activation in this data config";
            break;
        }

    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        switch (param.active) {
        //x > 0 ? x : 0
        case Active_relu:
            ker_act_fwd_int8_to_fp32<ReluDev>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
                    (float*)out_data, (const char*)in_data,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    in_shape[1] * in_shape[2] * in_shape[3],
                    in_shape[2] * in_shape[3], in_shape[3], 1,
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    negative_slope, coef, in_scale, count);
            break;
        case Active_sigmoid:
            ker_act_fwd_int8_to_fp32<SigmoidDev>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
                    (float*)out_data, (const char*)in_data,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    in_shape[1] * in_shape[2] * in_shape[3],
                    in_shape[2] * in_shape[3], in_shape[3], 1,
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    negative_slope, coef, in_scale, count);
            break;
        default:
            LOG(FATAL) << "Not implement this activation in this data config";
            break;
        }
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_INT8) {
        switch (param.active) {
        //x > 0 ? x : 0
        case Active_relu:
            ker_relu_fwd<char>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    (char*)out_data, (const char*)in_data, count, negative_slope,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        case Active_sigmoid:
            ker_sigmoid_fwd_int8
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                            (char*)out_data, (const char*)in_data, count,
                            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        default:
            LOG(FATAL) << "Not implement this activation in this data config";
            break;
        }

    } else if (inputs[0]->get_dtype() == AK_FLOAT && outputs[0]->get_dtype() == AK_FLOAT) {
        switch (param.active) {
        //x > 0 ? x : 0
        case Active_relu:
            ker_relu_fwd<float>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    (float*)out_data, (const float*)in_data, count, negative_slope,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        case Active_sigmoid:
            ker_sigmoid_fwd<float>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    (float*)out_data, (const float*)in_data, count,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3]);
            break;
        default:
            LOG(FATAL) << "Not implement this activation in this data config";
            break;
        }
    } else {
        LOG(FATAL) << "wrong data type config!!!";
    }

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberActivation<NV, AK_FLOAT>;
template class SaberActivation<NV, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, NV, AK_HALF);
}
}
