#include "saber/core/tensor_op.h"

#ifdef USE_BM

#include <random>

namespace anakin{

namespace saber{

typedef Tensor<BM>::API API;

template<>
void fill_tensor_rand<BM>(Tensor<BM>& tensor, \
    typename API::stream_t stream = NULL) {

    DataType type = tensor.get_dtype();
    switch (type){
        case AK_FLOAT: {
            float *host_mem_input = new float[tensor.size()];
            for (int i = 0; i < tensor.size(); ++i) {
                host_mem_input[i] = static_cast<float>(rand());
            }

            bm_device_mem_t* device_data_ptr = (bm_device_mem_t*) tensor.mutable_data();
            BMDNN_CHECK(bm_memcpy_s2d(API::get_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

            delete [] host_mem_input;
            break;
        }
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template<>
void fill_tensor_rand<BM>(Tensor<BM>& tensor, float vstart, \
    float vend, typename Tensor<BM>::API::stream_t stream = NULL){

    DataType type = tensor.get_dtype();
    switch (type){
        case AK_FLOAT: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0, 1.f);

            float *host_mem_input = new float[tensor.size()];
            for (int i = 0; i < tensor.size(); ++i) {
                float random_num = vstart + (vend - vstart) * dis(gen);
                host_mem_input[i] = random_num;
            }

            bm_device_mem_t* device_data_ptr = (bm_device_mem_t*) tensor.mutable_data();
            BMDNN_CHECK(bm_memcpy_s2d(API::get_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

            delete [] host_mem_input;
            break;
        }
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template<>
void fill_tensor_const<BM>(Tensor<BM>& tensor, float value, \
    typename Tensor<BM>::API::stream_t stream = NULL){

    DataType type = tensor.get_dtype();
    switch (type){
        case AK_FLOAT: {
            float *host_mem_input = new float[tensor.size()];
            for (int i = 0; i < tensor.size(); ++i) {
                host_mem_input[i] = value;
            }

            bm_device_mem_t* device_data_ptr = (bm_device_mem_t*) tensor.mutable_data();
            BMDNN_CHECK(bm_memcpy_s2d(API::get_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

            delete [] host_mem_input;
            break;
        }
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <>
void print_tensor<BM>(Tensor<BM>& tensor,  \
    typename Tensor<BM>::API::stream_t stream = NULL) {

    DataType type = tensor.get_dtype();
    switch (type){
        case AK_FLOAT: {
            LOG(INFO) << "BM device tensor data:" << tensor.size();

            /*
            const bm_device_mem_t* device_data_ptr = tensor.data();
            unsigned long long gaddr = bm_mem_get_device_addr(*device_data_ptr);
            bm_flush(get_bm_handle());
            float* device_data = (float*)bm_get_global_addr(gaddr);

            for (int i = 0; i < tensor.size(); ++i) {
                printf("%.2f ", device_data[i]);

                if ((i + 1) % (4 * tensor.width()) == 0) {
                    printf("\n");
                }
            }*/

            float *host_mem = new float[tensor.size()];
            auto* device_data_ptr = const_cast<bm_device_mem_t *>((bm_device_mem_t*) tensor.data());
            bm_memcpy_d2s(API::get_handle(), bm_mem_from_system(host_mem), *device_data_ptr);

            for (int i = 0; i < tensor.size(); ++i) {
                printf("%.2f\t", host_mem[i]);

                if ((i + 1) % tensor.width() == 0){
                    printf("\n");
                }
            }
            printf("\n");

            delete [] host_mem;
            break;
        }
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}



} //namespace saber

} //namespace anakin

#endif //USE_BM