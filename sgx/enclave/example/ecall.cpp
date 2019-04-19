#include "anakin_config.h"

#include <algorithm>
#include "stdio.h"

#include "graph.h"
#include "net.h"
#include "saber/core/tensor_op.h"
#include "mkl.h"

#include <sgx_tseal.h>

namespace {

using namespace anakin;

std::unique_ptr<graph::Graph<X86, Precision::FP32>> ModelGraph;
std::unique_ptr<Net<X86, Precision::FP32>> ModelNet;

}

namespace anakin {

extern "C" int setup_model(const char *model_name) {
    ModelGraph.reset(new graph::Graph<X86, Precision::FP32>());
    ModelGraph->load(model_name);
#ifdef ENABLE_DEBUG
    printf("model loaded\n");
#endif

    ModelGraph->Optimize();
#ifdef ENABLE_DEBUG
    printf("model optimized\n");
#endif

    ModelNet.reset(new Net<X86, Precision::FP32>(*ModelGraph, true));

    return 0;
}

extern "C" int seal_data(size_t input_size, const void *input,
                         size_t output_max_size, void *output,
                         size_t *result_size) {
    uint32_t output_len = sgx_calc_sealed_data_size(0, input_size);

    if (output_len > output_max_size) return -1;

    auto rc = sgx_seal_data(0, NULL, input_size, static_cast<const uint8_t *>(input),
                            output_len, static_cast<sgx_sealed_data_t *>(output));

    if (rc != SGX_SUCCESS) return -2;

    *result_size = output_len;

    return 0;
}

extern "C" int unseal_data(size_t input_size, const void *input,
                           size_t output_max_size, void *output,
                           size_t *result_size) {
    auto sealed_data = static_cast<const sgx_sealed_data_t *>(input);
    uint32_t input_len = sgx_get_encrypt_txt_len(sealed_data);

    if (input_len > output_max_size) return -1;

    uint32_t mac_length = 0;
    auto rc = sgx_unseal_data(sealed_data, NULL, &mac_length,
                              static_cast<uint8_t *>(output), &input_len);

    if (rc != SGX_SUCCESS) return -2;

    *result_size = input_len;

    return 0;
}

extern "C" int infer(size_t input_size, const void *input,
                     size_t output_max_size, void *output,
                     size_t *result_size) {

    if (!ModelNet) return -1;

    // Check input size requirement
    if (input_size != 0) {
        auto h_in = ModelNet->get_in_list().at(0);
        auto input_tensor_size = h_in->get_dtype_size() * h_in->valid_size();
        if (input_size != input_tensor_size) return -2;
    }
    
    // Check output size requirement
    auto h_out = ModelNet->get_out_list().at(0);
    auto output_tensor_size = h_out->get_dtype_size() * h_out->valid_size();
    if (output_tensor_size > output_max_size) return -3;

    if (input_size == 0) {
        for (auto h_in : ModelNet->get_in_list()) {
            fill_tensor_const(*h_in, 1);
        }
    } else {
        auto start = static_cast<const float *>(input);
        for (auto h_in : ModelNet->get_in_list()) {
            auto end = start + h_in->valid_size();
            std::copy(start, end, static_cast<float *>(h_in->data()));
            start = end;
        }
    }

    ModelNet->prediction();
    mkl_free_buffers();

    auto p_float = static_cast<const float *>(h_out->data());

#ifdef ENABLE_DEBUG
    auto c = h_out->valid_size();
    for (int i = 0; i < c; i++) {
        float f = p_float[i];
        printf("%f\n", f);
    }
#endif

    std::copy(p_float, p_float + h_out->valid_size(), static_cast<float *>(output));

    *result_size = output_tensor_size;

    return 0;
}

}
