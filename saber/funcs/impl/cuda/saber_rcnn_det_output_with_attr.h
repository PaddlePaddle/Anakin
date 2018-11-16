/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
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
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RCNN_DET_OUTPUT_WITH_ATTR_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RCNN_DET_OUTPUT_WITH_ATTR_H
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/funcs/impl/impl_rcnn_det_output_with_attr.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberRCNNDetOutputWithAttr<NV, OpDtype>: public ImplROIOutputSSD <
    NV, OpDtype > {
public:
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;


    SaberRCNNDetOutputWithAttr()
        : _img_info_data_host_tensor(NULL)
        , _rois_st_host_tensor(NULL)
        , _kpts_reg_st_host_tensor(NULL)
        , _kpts_exist_st_host_tensor(NULL)
        , _atrs_reg_st_host_tensor(NULL)
        , _ftrs_st_host_tensor(NULL)
        , _spmp_st_host_tensor(NULL)
        , _cam3d_st_host_tensor(NULL)
    {}
    ~SaberRCNNDetOutputWithAttr() {
        if (_img_info_data_host_tensor != NULL) {
            delete _img_info_data_host_tensor;
        }

        if (_rois_st_host_tensor != NULL) {
            delete _rois_st_host_tensor;
        }

        if (_kpts_reg_st_host_tensor != NULL) {
            delete _kpts_reg_st_host_tensor;
        }

        if (_kpts_exist_st_host_tensor != NULL) {
            delete _kpts_exist_st_host_tensor;
        }

        if (_atrs_reg_st_host_tensor != NULL) {
            delete _atrs_reg_st_host_tensor;
        }

        if (_ftrs_st_host_tensor != NULL) {
            delete _ftrs_st_host_tensor;
        }

        if (_spmp_st_host_tensor != NULL) {
            delete _spmp_st_host_tensor;
        }

        if (_cam3d_st_host_tensor != NULL) {
            delete _cam3d_st_host_tensor;
        }
    }
    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             ProposalParam<NV>& param, Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               ProposalParam<NV>& param, Context<NV>& ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 ProposalParam<NV>& param);
private:
    bool has_img_info_;
    int num_rois_;
    int rois_dim_;
    int num_kpts_;
    int kpts_cls_dim_;
    int kpts_reg_dim_;
    int num_atrs_;
    int num_ftrs_;
    int num_spmp_;
    int num_cam3d_;
    Tensor<NVHX86>* _img_info_data_host_tensor;
    Tensor<NVHX86>* _rois_st_host_tensor;
    Tensor<NVHX86>* _kpts_reg_st_host_tensor;
    Tensor<NVHX86>* _kpts_exist_st_host_tensor;
    Tensor<NVHX86>* _atrs_reg_st_host_tensor;
    Tensor<NVHX86>* _ftrs_st_host_tensor;
    Tensor<NVHX86>* _spmp_st_host_tensor;
    Tensor<NVHX86>* _cam3d_st_host_tensor;
};
}
}
#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H