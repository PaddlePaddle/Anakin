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
#include "include/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"


namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DetectionOutputParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DetectionOutputParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count = outputs[0]->valid_size();

    //! inputs[1]: confidence map, dims = 4 {N, classes * boxes, 1, 1}
    //! inputs[2]: prior boxes, dims = 4 {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
    Shape sh_loc  = inputs[0]->valid_shape();
    Shape sh_conf = inputs[1]->valid_shape();
    Shape sh_box  = inputs[2]->valid_shape();
    // Tensor<AMD> t_conf = inputs[1];
    //! shape {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
    //! layout must be 4 dims, the priors is in the last dim
    _num_priors = sh_box.count() / 8;
    int num     = inputs[0]->num();

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

    CHECK_EQ(_num_priors * _num_loc_classes * 4, sh_loc[1])
            << "Number of priors must match number of location predictions.";
    CHECK_EQ(_num_priors * _num_classes, sh_conf[1])
            << "Number of priors must match number of confidence predictions.";

    if (_conf_cpu_data != nullptr) {
        fast_free(_conf_cpu_data);
    }

    if (_bbox_cpu_data != nullptr) {
        fast_free(_bbox_cpu_data);
    }

    _conf_cpu_data = (dtype*)fast_malloc(sizeof(dtype) * sh_conf.count());
    _bbox_cpu_data = (dtype*)fast_malloc(sizeof(dtype) * sh_loc.count());

    const int loc_count = _bbox_preds.valid_size();

    AMDKernelPtr kptr;
    _kernels_ptr.clear();

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.kernel_type = SABER;
    kernelInfo.g_wk        = {(loc_count / 4 + 256 - 1) / 256 * 256};
    kernelInfo.kernel_file = "Detection.cl";

    if (param.type == CORNER) {
        if (param.variance_encode_in_target) {
            kernelInfo.kernel_name = "decode_bbox_corner_variance_kernel";

        } else {
            kernelInfo.kernel_name = "decode_bbox_corner_no_variance_kernel";
        }
    } else if (param.type == CENTER_SIZE) {
        if (param.variance_encode_in_target) {
            kernelInfo.kernel_name = "decode_bbox_center_variance_kernel";

        } else {
            kernelInfo.kernel_name = "decode_bbox_center_no_variance_kernel";
        }
    } else if (param.type == CORNER_SIZE) {
        if (param.variance_encode_in_target) {
            kernelInfo.kernel_name = "decode_bbox_corner_size_variance_kernel";
        } else {
            kernelInfo.kernel_name = "decode_bbox_corner_size_no_variance_kernel";
        }
    }

    kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    _kernels_ptr.push_back(kptr);

    if (!param.share_location) {
        kernelInfo.kernel_name = "permute_data_kernel";
        kernelInfo.g_wk        = {(loc_count + 256 - 1) / 256 * 256};

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    }

    kernelInfo.kernel_name = "permute_data_kernel";
    kernelInfo.g_wk = {(inputs[1]->valid_size() + 256 - 1) / 256 * 256};
    kptr            = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    _kernels_ptr.push_back(kptr);

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DetectionOutputParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    const int num = inputs[0]->num();

    // Decode predictions.
    const int loc_count = _bbox_preds.valid_size();

    int count = loc_count / 4;

    int j                  = 0; // kernel index
    int share_location_tmp = 0;

    if (param.share_location) {
        share_location_tmp = 1;
    }

    bool err = false;

    amd_kernel_list list;

    // decode_bboxes
    if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    err = _kernels_ptr[j].get()->SetKernelArgs(
              (PtrDtype)inputs[0]->data(),
              (PtrDtype)inputs[2]->data(),
              (int)_num_priors,
              (int)share_location_tmp,
              (int)_num_loc_classes,
              (int)param.background_id,
              (PtrDtype)_bbox_preds.mutable_data());
    list.push_back(_kernels_ptr[j]);
    j++;

    // Retrieve all decoded location predictions.
    if (!param.share_location) {
        int new_dim = 4;
        err         = _kernels_ptr[j].get()->SetKernelArgs(
                          (PtrDtype)_bbox_preds.mutable_data(),
                          (int)loc_count,
                          (int)_num_loc_classes,
                          (int)_num_priors,
                          (int)new_dim,
                          (PtrDtype)_bbox_permute.mutable_data());
        list.push_back(_kernels_ptr[j]);
        j++;
    }

    // Retrieve all confidences.
    int new_dim = 1;
    err         = _kernels_ptr[j].get()->SetKernelArgs(
                      (PtrDtype)inputs[1]->data(),
                      (int)inputs[1]->valid_size(),
                      (int)this->_num_classes,
                      (int)_num_priors,
                      (int)new_dim,
                      (PtrDtype)_conf_permute.mutable_data());
    list.push_back(_kernels_ptr[j]);
    j++;

    if (!err) {
        LOG(ERROR) << "Failed to set kernel args";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Failed to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";

    AMD_API::async_memcpy(
        _bbox_cpu_data,
        0,
        0,
        _bbox_preds.data(),
        0,
        inputs[0]->device_id(),
        _bbox_preds.valid_size() * sizeof(dtype),
        cm,
        __DtoH());
    AMD_API::async_memcpy(
        _conf_cpu_data,
        0,
        0,
        _conf_permute.data(),
        0,
        inputs[0]->device_id(),
        _conf_permute.valid_size() * sizeof(dtype),
        cm,
        __DtoH());
    AMD_API::sync_stream(NULL, cm);

    std::vector<dtype> result;

    nms_detect(_bbox_cpu_data, _conf_cpu_data, result, num, this->_num_classes, _num_priors,
               param.background_id, \
               param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta,
               param.share_location);

    if (result.size() == 0) {
        result.resize(7);

        for (int i = 0; i < 7; ++i) {
            result[i] = (dtype) - 1;
        }

        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, result.size() / 7, 7}));
    }

    AMD_API::async_memcpy(
        outputs[0]->mutable_data(),
        0,
        inputs[0]->device_id(),
        result.data(),
        0,
        0,
        result.size() * sizeof(dtype),
        cm,
        __HtoD());

    return SaberSuccess;
}
template class SaberDetectionOutput<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDetectionOutput, DetectionOutputParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberDetectionOutput, DetectionOutputParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
