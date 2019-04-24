#include "saber/funcs/impl/arm/saber_argmax.h"
#include "saber/funcs/type_trans.h"

namespace anakin{

namespace saber{
template <typename dtype>
void argmax_axis_kernel(const dtype* din, dtype* dout, const int topk, const bool has_max_val, \
    int size, int in_channel, int out_channel, int in_stride, int out_stride) {
    for (int n = 0; n < out_stride; n++){
        for (int k = 0; k < in_stride; k++){
            const dtype* din_ch = din + n * in_channel + k;
            std::vector< std::pair<dtype, int> > vec;
            vec.resize(size);
            for (int i = 0; i < size; i++){
                vec[i] = std::make_pair(din_ch[i * in_stride], i);
            }
            //sort
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), \
                std::greater< std::pair<dtype, int> >());
            //out
            dtype* dout_ch = dout + n * out_channel + k;
            if (has_max_val){
                for (int i = 0; i < topk; i++){
                    dout_ch[i * in_stride] = vec[i].first;
                }
            }else{
                for (int i = 0; i < topk; i++){
                    dout_ch[i * in_stride] = vec[i].second;
                }
            }
        }
    }
}

template <typename dtype>
void argmax_kernel(const dtype* din, dtype* dout, const int topk, const bool has_max_val, \
    int num, int in_channel, int out_channel) {
    for (int n = 0; n < num; n++){
        const dtype* din_ch = din + n * in_channel;
        std::vector< std::pair<dtype, int> > vec;
        vec.resize(in_channel);
        for (int i = 0; i < in_channel; i++){
            vec[i] = std::make_pair(din_ch[i], i);
        }
        //sort
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), \
            std::greater< std::pair<dtype, int> >());
        //out
        if (has_max_val){
            dtype* dout_ch = dout + n * out_channel;
            dtype* dout_index = dout_ch;
            dtype* dout_data = dout_ch + topk;
            for (int i = 0; i < topk; i++){
                dout_data[i] = vec[i].first;
                dout_index[i] = vec[i].second;
                //LOG(INFO) << "max_data: " <<dout_data[i] << ", max_index: "<<dout_index[i];
            }
        }else{
            dtype* dout_data = dout + n * out_channel;
            for (int i = 0; i < topk; i++){
                dout_data[i] = vec[i].second;
                // LOG(INFO) << "max_data: " <<vec[i].first << ", max_index: "<< dout_data[i];
            }
        }
        vec.clear();
    }
}
template <>
SaberStatus SaberArgmax<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ArgmaxParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int input_size = inputs.size();
    //! get output data, valid shape and stride shape
    int offset_Argmax_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();
    const int axis = param.axis;
    const int topk = param.top_k;
    const bool has_axis = param.has_axis;
    const bool has_max_val = param.out_max_val;

    DataType tensor_out_type = outputs[0]->get_dtype();
    const float* din = nullptr;
    float* dout = nullptr;
    if (tensor_out_type == AK_INT8) {
        _tmp_out.set_dtype(AK_FLOAT);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = static_cast<float*>(_tmp_out.mutable_data());
    } else {
        dout = static_cast<float*>(outputs[0]->mutable_data());
    }

    for (int i = 0; i < inputs.size(); ++i) {
        DataType tensor_in_type = inputs[i]->get_dtype();
        if (tensor_in_type == AK_INT8) {
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[i]->valid_shape());
            trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(*inputs[i], _tmp_in, outputs[0]->get_scale()[0], 1.f, {1.f});
            din = static_cast<const float *>(_tmp_in.data());
        } else {
            din = static_cast<const float*>(inputs[i]->data());
        }
        if (has_axis){
            int in_channel = in_shape.count(axis, in_shape.dims());
            int out_channel = out_shape.count(axis, out_shape.dims());
            int size = in_shape[axis];
            argmax_axis_kernel<float>(din, dout, topk, has_max_val, size, in_channel, \
                out_channel, _in_stride, _out_stride);
        }else{
            int in_channel = in_shape[1] * in_shape[2] * in_shape[3];
            int out_channel = out_shape[1] * out_shape[2] * out_shape[3];
            argmax_kernel<float>(din, dout, topk, has_max_val, in_shape[0], in_channel, out_channel);
        }
    }
    if (tensor_out_type == AK_INT8) {
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], 1.f, {1.f});
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Argmax : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops =  2.f * inputs[0]->valid_size();
    ops.ts = ts;
    OpTimer::add_timer("Argmax", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
template <>
SaberStatus SaberArgmax<ARM, AK_INT8>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ArgmaxParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int input_size = inputs.size();
    //! get output data, valid shape and stride shape
    int offset_Argmax_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();
    const int axis = param.axis;
    const int topk = param.top_k;
    const bool has_axis = param.has_axis;
    const bool has_max_val = param.out_max_val;

    DataType tensor_out_type = outputs[0]->get_dtype();

    const int8_t* din = nullptr;
    int8_t* dout = nullptr;
    if (tensor_out_type == AK_FLOAT) {
        _tmp_out.set_dtype(AK_INT8);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = static_cast<int8_t*>(_tmp_out.mutable_data());
    } else {
        dout = static_cast<int8_t*>(outputs[0]->mutable_data());
    }
    for (int i = 0; i < inputs.size(); ++i){
        DataType tensor_in_type = inputs[i]->get_dtype();
        if (tensor_in_type == AK_FLOAT) {
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[i]->valid_shape());
            trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[i], _tmp_in, outputs[0]->get_scale()[0], 1.f, {1.f});
            din = static_cast<const int8_t*>(_tmp_in.data());
        } else {
            din = static_cast<const int8_t*>(inputs[i]->data());
        }
        if (has_axis){
            int in_channel = in_shape.count(axis, in_shape.dims());
            int out_channel = out_shape.count(axis, out_shape.dims());
            int size = in_shape[axis];
            argmax_axis_kernel<int8_t>(din, dout, topk, has_max_val, size, in_channel, \
                out_channel, _in_stride, _out_stride);
        }else{
            int in_channel = in_shape[1] * in_shape[2] * in_shape[3];
            int out_channel = out_shape[1] * out_shape[2] * out_shape[3];
            argmax_kernel<int8_t>(din, dout, topk, has_max_val, in_shape[0], in_channel, out_channel);
        }
    }
    if (tensor_out_type == AK_FLOAT) {
        trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], 1.f, {1.f});
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Argmax : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops =  2.f * inputs[0]->valid_size();
    ops.ts = ts;
    OpTimer::add_timer("Argmax", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, ARM, AK_HALF);

} //namespace anakin

} //namespace anakin
