#include "test_lite.h"
#include "op_param.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

TEST(TestSaberLite, test_param) {

    int chout = 64;
    int chin = 32;
    int kw = 3;
    int kh = 3;
    int group = 1;
    int stride_w = 1;
    int stride_h = 1;
    int dila_w = 1;
    int dila_h = 1;
    int pad_w = 1;
    int pad_h = 1;
    bool flag_bias = true;
    bool flag_relu = true;
    ActiveType act_type = Active_relu;

    PoolingType pool_type = Pooling_average_include_padding;
    bool flag_global = true;
    int pool_kw = 2;
    int pool_kh = 2;
    int pool_stride_w = 2;
    int pool_stride_h = 2;
    int pool_pad_w = 0;
    int pool_pad_h = 0;

    float* weights = new float[chout * chin * kw * kh];
    float* bias = new float[chout];
    weights[0] = 6.66f;
    weights[1] = 8.88f;
    bias[0] = 1.0f;

    int w_size = chout * chin * kw * kh;

    LOG(INFO) << "construct ConvAct2D param: ";
    LOG(INFO) << " weights_size = " << w_size;
    LOG(INFO) << " out_channels = " << chout;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad = " << pad_w;
    LOG(INFO) << " stride = " << stride_w;
    LOG(INFO) << " dilation = " << dila_w;
    LOG(INFO) << " kernel = " << kw;
    LOG(INFO) << "bias flag = " << (flag_bias? "true" : "false");
    LOG(INFO) << "relu flag = " << (flag_relu? "true" : "false");
    LOG(INFO) << "act type = " << ((act_type == Active_relu)? "relu" : "unkown");
    LOG(INFO) << "weights: " << weights[0] << ", " << weights[1] << ", bias: " << bias[0];

    ConvAct2DParam param(w_size, chout, group, kw, kh, stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, act_type, flag_relu, weights, bias);

    LOG(WARNING) << "init param:";
            LOG(INFO) << " weights_size = " << param._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param._conv_param._num_output;
            LOG(INFO) << " group = " << param._conv_param._group;
            LOG(INFO) << " pad = " << param._conv_param._pad_w;
            LOG(INFO) << " stride = " << param._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param._conv_param._weights[0] << ", " << param._conv_param._weights[1] << ", bias: " << param._conv_param._bias[0];

    ConvAct2DParam param1(param);
    LOG(WARNING) << "param copy constructor:";
            LOG(INFO) << " weights_size = " << param1._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param1._conv_param._num_output;
            LOG(INFO) << " group = " << param1._conv_param._group;
            LOG(INFO) << " pad = " << param1._conv_param._pad_w;
            LOG(INFO) << " stride = " << param1._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param1._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param1._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param1._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param1._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param1._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param1._conv_param._weights[0] << ", " << param1._conv_param._weights[1] << ", bias: " << param1._conv_param._bias[0];

    ConvAct2DParam param2;
    param2 = param;
    LOG(WARNING) << "param operator = constructor:";
            LOG(INFO) << " weights_size = " << param2._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param2._conv_param._num_output;
            LOG(INFO) << " group = " << param2._conv_param._group;
            LOG(INFO) << " pad = " << param2._conv_param._pad_w;
            LOG(INFO) << " stride = " << param2._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param2._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param2._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param2._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param2._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param2._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param2._conv_param._weights[0] << ", " << param2._conv_param._weights[1] << ", bias: " << param2._conv_param._bias[0];

    PoolParam param4(pool_type, flag_global, pool_kw, pool_kh, pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h);
    LOG(WARNING) << "pool param init:";
            LOG(INFO) << " pool type = " << ((param4._pool_type == Pooling_average_include_padding)? "Pooling_average_include_padding" : "unknown");
            LOG(INFO) << " flag_global = " << (param4._flag_global? "true" : "false");
            LOG(INFO) << " pad = " << param4._pool_pad_w;
            LOG(INFO) << " stride = " << param4._pool_stride_w;
            LOG(INFO) << " kernel = " << param4._pool_kw;

    ConvActPool2DParam param5(w_size, chout, group, kw, kh, stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, \
        act_type, flag_relu, \
        pool_type, flag_global, pool_kw, pool_kh, pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h, weights, bias);

    LOG(WARNING) << "ConvActPool2Dparam init:";
            LOG(INFO) << " weights_size = " << param5._conv_act_param._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param5._conv_act_param._conv_param._num_output;
            LOG(INFO) << " group = " << param5._conv_act_param._conv_param._group;
            LOG(INFO) << " pad = " << param5._conv_act_param._conv_param._pad_w;
            LOG(INFO) << " stride = " << param5._conv_act_param._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param5._conv_act_param._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param5._conv_act_param._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param5._conv_act_param._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param5._conv_act_param._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param5._conv_act_param._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param5._conv_act_param._conv_param._weights[0] << ", " << param5._conv_act_param._conv_param._weights[1] << ", bias: " << param5._conv_act_param._conv_param._bias[0];
            LOG(INFO) << " pool type = " << ((param5._pool_param._pool_type == Pooling_average_include_padding)? "Pooling_average_include_padding" : "unknown");
            LOG(INFO) << " flag_global = " << (param5._pool_param._flag_global? "true" : "false");
            LOG(INFO) << " pad = " << param5._pool_param._pool_pad_w;
            LOG(INFO) << " stride = " << param5._pool_param._pool_stride_w;
            LOG(INFO) << " kernel = " << param5._pool_param._pool_kw;

    ConvActPool2DParam param6(param2, param4);

            LOG(WARNING) << "ConvActPool2Dparam init from 2 struct:";
            LOG(INFO) << " weights_size = " << param6._conv_act_param._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param6._conv_act_param._conv_param._num_output;
            LOG(INFO) << " group = " << param6._conv_act_param._conv_param._group;
            LOG(INFO) << " pad = " << param6._conv_act_param._conv_param._pad_w;
            LOG(INFO) << " stride = " << param6._conv_act_param._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param6._conv_act_param._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param6._conv_act_param._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param6._conv_act_param._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param6._conv_act_param._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param6._conv_act_param._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param6._conv_act_param._conv_param._weights[0] << ", " << param6._conv_act_param._conv_param._weights[1] << ", bias: " << param6._conv_act_param._conv_param._bias[0];
            LOG(INFO) << " pool type = " << ((param6._pool_param._pool_type == Pooling_average_include_padding)? "Pooling_average_include_padding" : "unknown");
            LOG(INFO) << " flag_global = " << (param6._pool_param._flag_global? "true" : "false");
            LOG(INFO) << " pad = " << param6._pool_param._pool_pad_w;
            LOG(INFO) << " stride = " << param6._pool_param._pool_stride_w;
            LOG(INFO) << " kernel = " << param6._pool_param._pool_kw;

    ConvActPool2DParam param7(param6);

    LOG(WARNING) << "ConvActPool2Dparam copy constructor:";
            LOG(INFO) << " weights_size = " << param7._conv_act_param._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param7._conv_act_param._conv_param._num_output;
            LOG(INFO) << " group = " << param7._conv_act_param._conv_param._group;
            LOG(INFO) << " pad = " << param7._conv_act_param._conv_param._pad_w;
            LOG(INFO) << " stride = " << param7._conv_act_param._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param7._conv_act_param._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param7._conv_act_param._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param7._conv_act_param._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param7._conv_act_param._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param7._conv_act_param._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param7._conv_act_param._conv_param._weights[0] << ", " << param7._conv_act_param._conv_param._weights[1] << ", bias: " << param7._conv_act_param._conv_param._bias[0];
            LOG(INFO) << " pool type = " << ((param7._pool_param._pool_type == Pooling_average_include_padding)? "Pooling_average_include_padding" : "unknown");
            LOG(INFO) << " flag_global = " << (param7._pool_param._flag_global? "true" : "false");
            LOG(INFO) << " pad = " << param7._pool_param._pool_pad_w;
            LOG(INFO) << " stride = " << param7._pool_param._pool_stride_w;
            LOG(INFO) << " kernel = " << param7._pool_param._pool_kw;


    ConvActPool2DParam param8;
    param8 = param6;

    LOG(WARNING) << "ConvActPool2Dparam operator= constructor:";
            LOG(INFO) << " weights_size = " << param8._conv_act_param._conv_param._weights_size;
            LOG(INFO) << " out_channels = " << param8._conv_act_param._conv_param._num_output;
            LOG(INFO) << " group = " << param8._conv_act_param._conv_param._group;
            LOG(INFO) << " pad = " << param8._conv_act_param._conv_param._pad_w;
            LOG(INFO) << " stride = " << param8._conv_act_param._conv_param._stride_w;
            LOG(INFO) << " dilation = " << param8._conv_act_param._conv_param._dila_w;
            LOG(INFO) << " kernel = " << param8._conv_act_param._conv_param._kw;
            LOG(INFO) << "bias flag = " << (param8._conv_act_param._conv_param._bias_term? "true" : "false");
            LOG(INFO) << "relu flag = " << (param8._conv_act_param._flag_act? "true" : "false");
            LOG(INFO) << "act type = " << ((param8._conv_act_param._act_type == Active_relu)? "relu" : "unkown");
            LOG(INFO) << "weights: " << param8._conv_act_param._conv_param._weights[0] << ", " << param8._conv_act_param._conv_param._weights[1] << ", bias: " << param8._conv_act_param._conv_param._bias[0];
            LOG(INFO) << " pool type = " << ((param8._pool_param._pool_type == Pooling_average_include_padding)? "Pooling_average_include_padding" : "unknown");
            LOG(INFO) << " flag_global = " << (param8._pool_param._flag_global? "true" : "false");
            LOG(INFO) << " pad = " << param8._pool_param._pool_pad_w;
            LOG(INFO) << " stride = " << param8._pool_param._pool_stride_w;
            LOG(INFO) << " kernel = " << param8._pool_param._pool_kw;

    delete [] weights;
    delete[] bias;
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

