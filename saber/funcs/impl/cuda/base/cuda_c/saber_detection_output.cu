#include "saber/funcs/impl/cuda/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"
namespace anakin{

namespace saber{
template <typename dtype>
__global__ void permute_data_kernel(const int nthreads,
                                  const dtype* data, const int num_classes, const int num_data,
                                  const int num_dim, dtype* new_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index % num_dim;
        const int c = (index / num_dim) % num_classes;
        const int d = (index / num_dim / num_classes) % num_data;
        const int n = index / num_dim / num_classes / num_data;
        const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
        new_data[new_index] = data[index];
    }
}

template <typename dtype>
void permute_data(const int nthreads,
                    const dtype* data, const int num_classes, const int num_data,
                    const int num_dim, dtype* new_data, cudaStream_t stream) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    permute_data_kernel<dtype><<<CUDA_GET_BLOCKS(nthreads),
            CUDA_NUM_THREADS, 0, stream>>>(nthreads, data, num_classes, num_data, num_dim, new_data);
}

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<NV, OpDtype>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    DetectionOutputParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    Tensor<NV>* t_loc = inputs[0];
    Tensor<NV>* t_conf = inputs[1];
    Tensor<NV>* t_prior = inputs[2];

    const dtype* loc_data = static_cast<const dtype*>(t_loc->data());
    const dtype* prior_data = static_cast<const dtype*>(t_prior->data());
    const int num = t_loc->num();

    // Decode predictions.
    dtype* bbox_data = static_cast<dtype*>(_bbox_preds.mutable_data());
    const int loc_count = _bbox_preds.valid_size();
    decode_bboxes<dtype>(loc_count, loc_data, prior_data, param.type, \
        param.variance_encode_in_target, _num_priors, param.share_location, \
        _num_loc_classes, param.background_id, bbox_data, stream);
    // Retrieve all decoded location predictions.
    if (!param.share_location) {
        dtype * bbox_permute_data = static_cast<dtype*>(_bbox_permute.mutable_data());
        permute_data<dtype>(loc_count, bbox_data, _num_loc_classes, _num_priors,
                              4, bbox_permute_data, stream);
    }
    // Retrieve all confidences.
    dtype* conf_permute_data = static_cast<dtype*>(_conf_permute.mutable_data());
    permute_data<dtype>(t_conf->valid_size(), static_cast<dtype*>(t_conf->data()), \
         this->_num_classes, _num_priors, 1, conf_permute_data, stream);

    CUDA_CHECK(cudaMemcpyAsync(_bbox_cpu_data, static_cast<dtype*>(_bbox_preds.data()), \
                _bbox_preds.valid_size() * sizeof(dtype), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(_conf_cpu_data, static_cast<dtype*>(_conf_permute.data()), \
                _conf_permute.valid_size() * sizeof(dtype), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::vector<dtype> result;

    nms_detect(_bbox_cpu_data, _conf_cpu_data, result, num, this->_num_classes, _num_priors, param.background_id, \
        param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, param.share_location);

    if(result.size() == 0) {
        result.resize(7);
        for (int i = 0; i < 7; ++i) {
            result[i] = (dtype)-1;
        }
        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, static_cast<int>(result.size() / 7), 7}));
    }

    CUDA_CHECK(cudaMemcpyAsync(outputs[0]->mutable_data(), result.data(), \
                result.size() * sizeof(dtype), cudaMemcpyHostToDevice, stream));

    return SaberSuccess;
}

//template class SaberDetectionOutput<AK_FLOAT, NCHW>;
} //namespace anakin

} //namespace anakin
