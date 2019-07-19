#include "saber/funcs/impl/x86/saber_argmax.h"

namespace anakin {

namespace saber {


template <typename dtype>
void Argmax_kernel_axis(const dtype* din, dtype* dout, int num, int in_stride, \
                        int out_stride, int size, int in_ss, int out_ss, int top, bool out_max) {
    for (int n = 0; n < num * out_stride; n++) {
        for (int k = 0; k < in_stride; k ++) {
            const dtype* din_ch = din + n * in_ss + k;
            std::vector< std::pair<dtype, int> > vec;
            vec.resize(size);

            for (int i = 0; i < size; i++) {
                vec[i] = std::make_pair(din_ch[i * in_stride], i);
            }

            //sort
            std::partial_sort(vec.begin(), vec.begin() + top, vec.end(),
                              std::greater< std::pair<float, int> >());
            //out
            dtype* dout_ch = dout + n * out_ss + k;

            for (int i = 0; i < top ; i ++) {
                if (out_max) {
                    dout_ch[i * in_stride] = vec[i].first;
                } else {
                    dout_ch[i * in_stride] = vec[i].second;
                }
            }
        }
    }
}

template <typename dtype>
void Argmax_kernel(const dtype* din, dtype* dout, int num, int in_channel, \
                   int out_channel, int top, bool out_max) {
    for (int n = 0; n < num; n++) {
        const dtype* din_ch = din + n * in_channel;
        std::vector< std::pair<dtype, int> > vec;
        vec.resize(in_channel);

        for (int i = 0; i < in_channel; i++) {
            vec[i] = std::make_pair(din_ch[i], i);
        }

        //sort
        std::partial_sort(vec.begin(), vec.begin() + top, vec.end(),
                          std::greater< std::pair<float, int> >());

        //out
        if (out_max) {
            dtype* dout_ch = dout + n * out_channel;
            dtype* dout_index = dout_ch;
            dtype* dout_data = dout_ch + top;

            for (int i = 0; i < top; i++) {
                dout_data[i] = vec[i].first;
                dout_index[i] = vec[i].second;
                //LOG(INFO) << "max_data: " <<dout_data[i] << ", max_index: "<<dout_index[i];
            }
        } else {
            dtype* dout_data = dout + n * out_channel;

            for (int i = 0; i < top; i++) {
                dout_data[i] = vec[i].second;
                // LOG(INFO) << "max_data: " <<vec[i].first << ", max_index: "<< dout_data[i];
            }
        }

        //vec.clear();
    }
}

template <DataType OpDtype>
SaberStatus SaberArgmax<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs, ArgmaxParam<X86>& param) {
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();

    int ch_out = outputs[0]->channel();
    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();

    int top = param.top_k;
    bool has_ax = param.has_axis;
    int ax = param.axis;
    bool out_max = param.out_max_val;

    const OpDataType* din = (const OpDataType*)inputs[0]->data();
    OpDataType* dout = (OpDataType*)outputs[0]->mutable_data();
    int in_channel = channel * height * width;
    int out_channel = ch_out * w_out * h_out;

    if (has_ax) { //nchw
        auto shape = inputs[0]->valid_shape();
        int stride = shape.count(ax + 1, shape.dims());
        int out_stride = shape.count(1, ax);
        int out_ss = outputs[0]->valid_shape().count(ax, shape.dims());
        int in_ss = shape.count(ax, shape.dims());
        // LOG(INFO) << "stride: "<<stride << ", out_stride: " << out_stride;
        int size = shape[ax];

        if (size < top) {
            LOG(INFO) << "input data size less than topk";
            return SaberUnImplError;
        }

        /*
        for (int n = 0; n < num * out_stride; n++){
            for(int k = 0; k < stride; k ++){
                const OpDataType* din_ch = din + n * in_ss + k;
                std::vector< std::pair<OpDataType, int> > vec;
                vec.resize(size);
                for (int i = 0; i < size; i++){
                    vec[i] = std::make_pair(din_ch[i*stride], i);
                }
                 //sort
                std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
                //out
                OpDataType* dout_ch = dout + n * out_ss + k;
                for(int i = 0; i < top ;i ++){
                    if(out_max)
                        dout_ch[i*stride] = vec[i].first;
                    else
                        dout_ch[i*stride] = vec[i].second;
                }
            }
        }
        */
        Argmax_kernel_axis<float>(din, dout, num, stride, out_stride, size, in_ss, out_ss, top, out_max);
    } else { //all
        if (in_channel < top) {
            LOG(INFO) << "input data size less than topk";
            return SaberUnImplError;
        }

        /*
        for (int n = 0; n < num; n++){
            const OpDataType* din_ch = din + n * in_channel;
            std::vector< std::pair<OpDataType, int> > vec;
            vec.resize(in_channel);
            for (int i = 0; i < in_channel; i++){
                vec[i] = std::make_pair(din_ch[i], i);
            }
            //sort
            std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
            //out
            if(out_max){
                OpDataType* dout_ch = dout + n * out_channel;
                OpDataType* dout_data = dout_ch;
                OpDataType* dout_index = dout_ch + top;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].first;
                    dout_index[i] = vec[i].second;
                    //LOG(INFO) << "max_data: " <<dout_data[i] << ", max_index: "<<dout_index[i];
                }
            }else{
                OpDataType* dout_data = dout + n * out_channel;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].second;
                   // LOG(INFO) << "max_data: " <<vec[i].first << ", max_index: "<< dout_data[i];
                }
            }
            vec.clear();
        }
        */
        Argmax_kernel<float>(din, dout, num, in_channel, out_channel, top, out_max);
    }

    return SaberSuccess;
}

template class SaberArgmax<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
