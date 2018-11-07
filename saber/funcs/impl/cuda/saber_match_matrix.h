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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MATCH_MATRIX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MATCH_MATRIX_H

#include "saber/funcs/impl/impl_match_matrix.h"
#include "saber/funcs/gemm.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberMatchMatrix<NV, OpDtype> :
    public ImplBase<
        NV, OpDtype,
        MatchMatrixParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    SaberMatchMatrix() : _handle(NULL) {};
    ~SaberMatchMatrix() {
        if (_handle) {
            cublasDestroy(_handle);
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            MatchMatrixParam<NV>& param, Context<NV>& ctx) {
        
        this->_ctx = &ctx;
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
        create(inputs, outputs, param, ctx);
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            MatchMatrixParam<NV>& param, Context<NV> &ctx) {
        if (&ctx != this->_ctx) {
            if (_handle != NULL) {
                CUBLAS_CHECK(cublasDestroy(_handle));
            }
            this->_ctx = &ctx;
            cudaStream_t cuda_stream;
            cuda_stream = ctx.get_compute_stream();
            CUBLAS_CHECK(cublasCreate(&_handle));
            CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
        }
        this->_ctx = &ctx;
        auto offset_l = inputs[0]->get_seq_offset()[0];
        auto offset_r = inputs[1]->get_seq_offset()[0];
        int batch = offset_r.size() - 1;
        int batch_word_r = offset_r[batch];
        int len_l = offset_l[1] - offset_l[0];
        int dim_t = param.dim_t;
        int dim_in = param.dim_in;
        int max_len_r = 0;
        for (int i = 0; i < offset_r.size() - 1; i++) {
            int cur_len = offset_r[i+1] - offset_r[i];
            max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
        }
        _input_l_transform.reshape(std::vector<int>{1, dim_t, dim_in, len_l});
        _input_l_transform_reorganize.reshape(std::vector<int>{1, dim_t, len_l, dim_in});
        _output_tmp.reshape(std::vector<int>{1, batch_word_r, dim_t, len_l});
        outputs[0]->reshape(std::vector<int>{batch, dim_t, len_l, max_len_r});
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          MatchMatrixParam<NV>& param);

protected:
    cublasHandle_t _handle;
    Tensor<NV> _input_l_transform;
    Tensor<NV> _input_l_transform_reorganize;
    Tensor<NV> _output_tmp;
    Tensor<NV> _offset_r;
    Gemm<NV, SABER_IMPL, float> _gemm_l_transform;
    Gemm<NV, SABER_IMPL, float> _gemm_r_transform;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MATCH_MATRIX_H
