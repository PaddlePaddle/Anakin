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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_LRN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_LRN_H

#include "saber/funcs/impl/impl_lrn.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderLrn<NV, OpDtype> : public ImplBase<
        NV, OpDtype, LrnParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    VenderLrn()
            : _handle(NULL)
            , _lrn_descs(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
            , _lrn_mode(CUDNN_LRN_CROSS_CHANNEL_DIM1)
    {}

    ~VenderLrn() {

        if (_lrn_descs) {
            CUDNN_CHECK(cudnnDestroyLRNDescriptor(_lrn_descs));
        }
        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
    }

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     param                [lrn parameters]
     */
    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             LrnParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               LrnParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 LrnParam<NV>& param);

private:
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnLRNDescriptor_t _lrn_descs;
    cudnnLRNMode_t _lrn_mode;

    const bool _use_tensor_core = true;

};

}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_LRN_H
