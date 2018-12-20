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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DETECTION_OUTPUT_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DETECTION_OUTPUT_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/impl/impl_detection_output.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDetectionOutput<AMD, OpDtype> :
    public ImplBase<AMD, OpDtype, DetectionOutputParam<AMD>> {

public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype dtype;
    typedef AMD_API::TPtr PtrDtype;

    SaberDetectionOutput() {
        _kernels_ptr.clear();
    }

    ~SaberDetectionOutput() {
        _kernels_ptr.clear();

        if (_bbox_cpu_data) {
            fast_free(_bbox_cpu_data);
        }

        if (_conf_cpu_data) {
            fast_free(_conf_cpu_data);
        }
    }

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         DetectionOutputParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           DetectionOutputParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        DetectionOutputParam<AMD>& param) override;

private:
    std::vector<AMDKernelPtr> _kernels_ptr;
    int _num_classes;
    int _num_loc_classes;
    int _num_priors;
    Tensor<AMD> _bbox_preds;
    Tensor<AMD> _bbox_permute;
    Tensor<AMD> _conf_permute;
    dtype* _bbox_cpu_data {nullptr};
    dtype* _conf_cpu_data {nullptr};
};

template class SaberDetectionOutput<AMD, AK_FLOAT>;

} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DETECTION_OUTPUT_H
