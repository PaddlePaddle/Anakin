/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_1X1_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_1X1_H

#include "saber/funcs/impl/impl_conv.h"
#include "saber/core/tensor.h"

namespace anakin {
namespace saber {

class BiasReluUtis {
public:
    BiasReluUtis() {

    }
    void reset(bool flag_bias, bool flag_relu, bool neg_relu) {
        if (flag_bias && flag_relu && neg_relu) {
            func = bias_relu<true, true, true>;
        } else if (flag_bias && flag_relu && !neg_relu) {
            func = bias_relu<true, true, false>;
        } else if (flag_bias && !flag_relu && !neg_relu) {
            func = bias_relu<true, false, false>;
        } else if (!flag_bias && flag_relu && neg_relu) {
            func = bias_relu<false, true, true>;
        } else if (!flag_bias && flag_relu && !neg_relu) {
            func = bias_relu<false, true, false>;
        } else if (!flag_bias && !flag_relu){
            func = bias_relu<false, false, false>;
        }else{
            LOG(FATAL) << "invalid init BiasReluUtis";
        }
    }

    void run(float* output, const float* bias, int batch_size, int out_c, int out_stride,
             float negative_slope) {

        func(output, bias, batch_size, out_c, out_stride, negative_slope);
    }


    template <bool flag_bias, bool flag_relu, bool neg_relu>
    static void bias_relu(float* output, const float* bias, int batch_size, int out_c, int out_stride,
                          float negative_slope) {
        int batch_stride = out_c * out_stride;
        if (flag_bias && !flag_relu) {
            #pragma omp parallel for collapse(3) schedule(static)

            for (int i = 0; i < batch_size; i++) {
                for (int oc = 0; oc < out_c; ++oc) {
                    for (int inner_id = 0; inner_id < out_stride; ++inner_id) {
                        int id = i * batch_stride + oc * out_stride + inner_id;
                        output[id] += bias[oc];
                    }
                }
            }
        } else if (!flag_bias && flag_relu) {
            #pragma omp parallel for collapse(3) schedule(static)

            for (int i = 0; i < batch_size; i++) {
                for (int oc = 0; oc < out_c; ++oc) {
                    for (int inner_id = 0; inner_id < out_stride; ++inner_id) {
                        int id = i * batch_stride + oc * out_stride + inner_id;

                        if (neg_relu) {
                            if (output[id] < 0.f) {
                                output[id] = output[id] * negative_slope;
                            }
                        } else {
                            if (output[id] < 0.f) {
                                output[id] = 0.f;
                            }
                        }
                    }
                }
            }
        } else if (flag_bias && flag_relu) {
            #pragma omp parallel for collapse(3) schedule(static)

            for (int i = 0; i < batch_size; i++) {
                for (int oc = 0; oc < out_c; ++oc) {
                    for (int inner_id = 0; inner_id < out_stride; ++inner_id) {
                        int id = i * batch_stride + oc * out_stride + inner_id;
                        float temp = output[id];
                        temp += bias[oc];

                        if (neg_relu) {
                            if (temp < 0.f) {
                                temp = temp * negative_slope;
                            }
                        } else {
                            if (temp < 0.f) {
                                temp = 0.f;
                            }
                        }

                        output[id] = temp;
                    }
                }
            }
        }
    }


private:
    std::function<void(float*, const float*, int, int, int, float)> func;
    //        void (*func)(float* output,const float* bias,int batch_size,int out_c, int out_stride,float negative_slope);

};

template <DataType OpDtype>
class SaberConv1X1: public ImplBase <
    X86, OpDtype, ConvEltwiseParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberConv1X1()
    {}

    ~SaberConv1X1() {
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             ConvEltwiseParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               ConvEltwiseParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ConvEltwiseParam<X86>& param);

private:
    BiasReluUtis _bias_utils;

    bool _flag_relu;
    bool _flag_neg;
    bool _flag_bias;
    float _neg_slope;

    int _out_c;
    int _in_c;
    int h;
    int w;
    int _in_inner_size;
    int _num_input;
    int _num_size_in;
    int _num_size_out;
    float _add_output;
    const OpDataType* _bias;
};


} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_H
