//
//
//#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
//#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
//#include "saber/funcs/impl/impl_lstm.h"
//#include "saber/funcs/impl/x86/x86_utils.h"
//namespace anakin {
//
//    namespace saber {
//
//        template<DataType OpDtype,
//                DataType inDtype,
//                DataType outDtype,
//                typename LayOutType_op,
//                typename LayOutType_in,
//                typename LayOutType_out>
//        class SaberLSTM<X86, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out> : \
//    public ImplBase <
//                Tensor<X86, inDtype, LayOutType_in>, \
//    Tensor<X86, outDtype, LayOutType_out>, \
//    Tensor<X86, OpDtype, LayOutType_op>, \
//    LSTMParam<Tensor<X86, OpDtype, LayOutType_op> >> {
//
//        public:
//            typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
//            typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
//            typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
//
//            typedef typename DataTensor_in::Dtype InDataType;
//            typedef typename DataTensor_out::Dtype OutDataType;
//            typedef typename OpTensor::Dtype OpDataType;
//
//            SaberLSTM() {}
//
//            ~SaberLSTM() {}
//
//            virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
//                             std::vector<DataTensor_out*>& outputs, \
//                             LSTMParam<OpTensor>& param, Context<X86>& ctx) {
//                this->_ctx=ctx;
//                if(param.with_peephole)
//                    _hidden_size = param.bias()->valid_size() / 7;
//                else
//                    _hidden_size = param.bias()->valid_size() / 4;
//
//                int weights_bias_size = _hidden_size * 4;
//                int weights_h2h_size = _hidden_size * _hidden_size * 3;
//                int weights_i2h_size = param.weight()->valid_size() - weights_h2h_size;
//                _word_size = weights_i2h_size / _hidden_size / 3;
//
//                _weights_i2h.try_expand_size(weights_i2h_size);
//                _weights_h2h.try_expand_size(weights_h2h_size);
//                _weights_bias.try_expand_size(weights_bias_size);
//
//                memcpy(_weights_i2h.mutable_data(), param.weight()->data(),
//                       sizeof(InDataType) * weights_i2h_size);
//                memcpy(_weights_h2h.mutable_data(), param.weight()->data() + weights_i2h_size,
//                       sizeof(InDataType) * weights_h2h_size);
//                memcpy(_weights_bias.mutable_data(), param.bias()->data(),
//                       sizeof(InDataType) * weights_bias_size);
//
//                        LOG(INFO)<<"success init";
//                return create(inputs,outputs,param,ctx);
//            }
//
//            virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
//                               std::vector<DataTensor_out*>& outputs, \
//                               LSTMParam<OpTensor>& param, Context<X86>& ctx) {
//
//
//                return SaberSuccess;
//            }
//
//
//            virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
//                                         std::vector<DataTensor_out*>& outputs,
//                                         LSTMParam<OpTensor>& param);
//
//        private:
//            int _word_size;
//            int _hidden_size;
//
//            bool _aligned_way=true;
//            int _aligned_word_size;
//            int _aligned_hidden_size;
//            int _aligned_size;
//            int _aligned_word_size_iter_num;
//            int _aligned_hidden_size_iter_num;
//
//            OpTensor _weights_i2h;
//            OpTensor _weights_h2h;
//            OpTensor _weights_bias;
//            DataTensor_out _init_hidden;
//
//            OpTensor _aligned_weights_i2h;
//            OpTensor _aligned_weights_h2h;
//            OpTensor _aligned_weights_bias;
//            DataTensor_out _aligned_init_hidden;
//
//            DataTensor_out _temp_wx;
//            DataTensor_out _temp_wh;
//            DataTensor_out _temp_whr;
//
//            DataTensor_in _temp_x;
//            DataTensor_out _temp_out;
//            DataTensor_out _temp_h_init;
//
//        };
//
//    }
//}
//#endif
