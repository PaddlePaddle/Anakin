
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LSTMP_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LSTMP_H
#include "saber/funcs/impl/impl_lstmp.h"
#include "sass_funcs.h"
namespace anakin {

namespace saber {

template<DataType OpDtype>
class SaberLstmp<NV, OpDtype> : public ImplBase <
    NV, OpDtype, LstmParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberLstmp() {}

    ~SaberLstmp() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs, \
                             std::vector<Tensor<NV> *>& outputs, \
                             LstmParam<NV>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;
        _inner_hidden_dim = param.cell_dim;
        _output_hidden_dim = param.project_dim;
        CHECK_GT(param.cell_dim,0);
        CHECK_GT(param.project_dim,0);

        CHECK_EQ(inputs.size(), 1) << "only support input size = 1";
        CHECK_EQ(outputs.size(), 1) << "only support outputs size = 1";
        CHECK_EQ(param.init_hidden() == nullptr, true) << "only support param.init_hidden() == nullptr";
        CHECK_EQ(param.num_layers, 1) << "only support param.num_layers==1";

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs, \
                               std::vector<Tensor<NV> *>& outputs, \
                               LstmParam<NV>& param, Context<NV>& ctx) {
        if (!(&ctx == this->_ctx)) {
            if (_handle != NULL) {
                CUBLAS_CHECK(cublasDestroy(_handle));
            }

            this->_ctx = &ctx;

            cudaStream_t cuda_stream;
            cuda_stream = ctx.get_compute_stream();
            CUBLAS_CHECK(cublasCreate(&_handle));
            CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
        }

        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 LstmParam<NV>& param);

private:

    cublasHandle_t _handle;

    Tensor<NV> _wx_tensor;
    Tensor<NV> _temp_hidden_tensor;
    Tensor<NV> _temp_cell_tensor;
    int _output_hidden_dim;
    int _inner_hidden_dim;


};
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LSTMP_H
