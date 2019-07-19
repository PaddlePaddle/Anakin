#include "saber/funcs/impl/cuda/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"
namespace anakin{

namespace saber{
template <typename dtype>
__global__ void permute_data_kernel(const int nthreads,
        const dtype* data, const int num_classes, const int priors,
        const int num_dim, dtype* new_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index % num_dim;
        const int c = (index / num_dim) % num_classes;
        const int d = (index / num_dim / num_classes) % priors;
        const int n = index / num_dim / num_classes / priors;
        const int new_index = ((n * num_classes + c) * priors + d) * num_dim + i;
        new_data[new_index] = data[index];
    }
}

template <typename dtype>
void permute_data(const int nthreads, const dtype* data, const int num_classes, const int priors, \
    const int num_dim, dtype* new_data, cudaStream_t stream) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    permute_data_kernel<dtype><<<CUDA_GET_BLOCKS(nthreads),
            CUDA_NUM_THREADS, 0, stream>>>(nthreads, data, num_classes, priors, num_dim, new_data);
}

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<NV, OpDtype>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    DetectionOutputParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    Tensor<NV>* t_loc = inputs[0];
    Tensor<NV>* t_conf = inputs[1];
    Tensor<NV>* t_prior;

    CHECK_EQ(t_loc->get_dtype(), AK_FLOAT) << "input data type must be float";
    CHECK_EQ(t_conf->get_dtype(), AK_FLOAT) << "input data type must be float";

    std::vector<int> priors;

    if (_shared_loc) {
        //! for one stage
        const int num = t_loc->num();
        for (int i = 0; i < num; ++i) {
            priors.push_back(_num_priors / num);
        }
        //! for ssd
        bool is_ssd = inputs.size() > 2;
        if (is_ssd) {
            t_prior = inputs[2];
        }
        if (is_ssd) {
            int num_priors = _num_priors / num;
            auto loc_data = static_cast<const float*>(t_loc->data());
            auto prior_data = static_cast<const float*>(t_prior->data());

            // Decode predictions.
            float* bbox_data = static_cast<float*>(_bbox_preds.mutable_data());
            const int loc_count = _bbox_preds.valid_size();
            decode_bboxes<float>(loc_count, loc_data, prior_data, param.type, \
            param.variance_encode_in_target, num_priors, param.share_location, \
            _num_loc_classes, param.background_id, bbox_data, stream);
            // Retrieve all decoded location predictions.
            if (!param.share_location) {
                float * bbox_permute_data = static_cast<float*>(_bbox_permute.mutable_data());
                permute_data<float>(loc_count, bbox_data, _num_loc_classes, num_priors,
                                    4, bbox_permute_data, stream);
            }
            // Retrieve all confidences.
            float* conf_permute_data = static_cast<float*>(_conf_permute.mutable_data());
            permute_data<float>(t_conf->valid_size(), static_cast<float*>(t_conf->data()), \
             this->_num_classes, num_priors, 1, conf_permute_data, stream);
            CUDA_CHECK(cudaMemcpyAsync(_bbox_cpu_data, static_cast<float*>(_bbox_preds.data()), \
                _bbox_preds.valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(_conf_cpu_data, static_cast<float*>(_conf_permute.data()), \
                _conf_permute.valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        } else { //! for multiclass nms
            CUDA_CHECK(cudaMemcpyAsync(_bbox_cpu_data, t_loc->data(), \
                t_loc->valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(_conf_cpu_data, t_conf->data(), \
                t_conf->valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        cudaStreamSynchronize(stream);
    } else {
        auto conf_permute = static_cast<float*>(_conf_permute.mutable_data());
        auto bbox_permute = static_cast<float*>(_bbox_permute.mutable_data());
        auto conf_ori = static_cast<const float*>(t_conf->data());
        auto bbox_ori = static_cast<const float*>(t_loc->data());
        //! for two stage
        //! sizeof seq offset is N + 1
        auto offset = t_loc->get_seq_offset()[0];
        for (int i = 0; i < offset.size() - 1; ++i) {
            int num_priors = offset[i + 1] - offset[i];
            priors.push_back(num_priors);
            const float* conf_ori_batch = conf_ori + this->_num_classes * offset[i];
            const float* bbox_ori_batch = bbox_ori + this->_num_classes * 4 * offset[i];
            float* conf_permute_batch = conf_permute + this->_num_classes * offset[i];
            float* bbox_permute_batch = bbox_permute + this->_num_classes * 4 * offset[i];
            //! permute conf and bbox
            //! input bbox layout is [M, C, 4], multi-batch view: [{priors0, C, 4}, {priors1, C, 4}, ...]
            //! permute bbox data to [{C, priors0, 4}, {C, priors1, 4}, ...]
            //! input conf layout is [M, C], multi-batch view: [{priors0, C}, {priors1, C}, ...]
            //! permute conf data to [{C, priors0}, {C, priors1}, ...]
            permute_data<float>(num_priors * this->_num_classes, conf_ori_batch,
                    this->_num_classes, num_priors, 1, conf_permute_batch, stream);
            permute_data<float>(num_priors * this->_num_classes * 4, bbox_ori_batch,
                    this->_num_classes, num_priors, 4, bbox_permute_batch, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(_bbox_cpu_data, bbox_permute, \
                _bbox_permute.valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(_conf_cpu_data, conf_permute, \
                _conf_permute.valid_size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    std::vector<float> result;
    nms_detect(_bbox_cpu_data, _conf_cpu_data, result, priors, this->_num_classes, param.background_id, \
        param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, _shared_loc);
    if(result.size() == 0) {
        result.resize(7);
        for (int i = 0; i < 7; ++i) {
            result[i] = (float)-1;
        }
        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, static_cast<int>(result.size() / 7), 7}));
    }

    CUDA_CHECK(cudaMemcpyAsync(outputs[0]->mutable_data(), result.data(), \
                result.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    return SaberSuccess;
}

//template class SaberDetectionOutput<AK_FLOAT, NCHW>;
} //namespace anakin

} //namespace anakin
