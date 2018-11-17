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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H

#include "saber/funcs/impl/impl_detection_output.h"
#include "saber/core/data_traits.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberDetectionOutput<NV, OpDtype> : \
    public ImplBase<
        NV, OpDtype,
        DetectionOutputParam<NV> > 
{
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype dtype;

    SaberDetectionOutput() = default;
    ~SaberDetectionOutput() {
        if (_bbox_cpu_data) {
            fast_free(_bbox_cpu_data);
        }
        if (_conf_cpu_data) {
            fast_free(_conf_cpu_data);
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            DetectionOutputParam<NV>& param, Context<NV>& ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            DetectionOutputParam<NV>& param, Context<NV> &ctx) {

        //! inputs[0]: location map, dims = 4 {N, boxes * 4, 1, 1}
        //! inputs[1]: confidence map, dims = 4 {N, classes * boxes, 1, 1}
        //! inputs[2]: prior boxes, dims = 4 {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
        Shape sh_loc = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();
        Shape sh_box = inputs[2]->valid_shape();
        //! shape {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
        //! layout must be 4 dims, the priors is in the last dim
        _num_priors = sh_box.count() / 8;
        int num = inputs[0]->num();
        if (param.class_num == 0) {
            _num_classes = inputs[1]->valid_size() / (num * _num_priors);
        } else {
            _num_classes = param.class_num;
        }
        if (param.share_location) {
            _num_loc_classes = 1;
        } else {
            _num_loc_classes = _num_classes;
            _bbox_permute.reshape(sh_loc);
        }

        _bbox_preds.reshape(sh_loc);
        _conf_permute.reshape(sh_conf);

        CHECK_EQ(_num_priors * _num_loc_classes * 4, sh_loc.count() / sh_loc.num()) << \
            "Number of priors must match number of location predictions.";
        CHECK_EQ(_num_priors * _num_classes, sh_conf.count() / sh_conf.num()) << \
            "Number of priors must match number of confidence predictions.";

        if (_conf_cpu_data != nullptr) {
            fast_free(_conf_cpu_data);
        }
        if (_bbox_cpu_data != nullptr) {
            fast_free(_bbox_cpu_data);
        }
        _conf_cpu_data = (dtype*)fast_malloc(sizeof(dtype) * sh_conf.count());
        _bbox_cpu_data = (dtype*)fast_malloc(sizeof(dtype) * sh_loc.count());

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV> *>& outputs,
                          DetectionOutputParam<NV>& param);


private:
    int _num_classes;
    int _num_loc_classes;
    int _num_priors;
    Tensor<NV> _bbox_preds;
    Tensor<NV> _bbox_permute;
    Tensor<NV> _conf_permute;
    dtype* _bbox_cpu_data{nullptr};
    dtype* _conf_cpu_data{nullptr};
};
template class SaberDetectionOutput<NV>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
