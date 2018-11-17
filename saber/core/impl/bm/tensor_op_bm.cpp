#include "core/tensor_op.h"
namespace anakin {

namespace saber {

template<>
void fill_tensor_const<BM>(Tensor<BM>& tensor, float value,
                           typename Tensor<BM>::API::stream_t stream = NULL) {
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_const(temp_tensor, value);
    tensor.copy_from(temp_tensor);
}
template<>
void fill_tensor_rand<BM>(Tensor<BM>& tensor, typename Tensor<BM>::API::stream_t stream = NULL) {
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_rand(temp_tensor);
    tensor.copy_from(temp_tensor);
}

template<>
void fill_tensor_rand<BM>(Tensor<BM>& tensor, float vstart, float vend,
                          typename Tensor<BM>::API::stream_t stream = NULL) {
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_rand(temp_tensor, vstart, vend);
    tensor.copy_from(temp_tensor);
}

template<>
void print_tensor<BM>(Tensor<BM>& tensor, typename Tensor<BM>::API::stream_t stream = NULL) {
    LOG(INFO) << "device tensor data";
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    print_tensor(temp_tensor);
}

template<>
void print_tensor_valid<BM>(Tensor<BM>& tensor, typename Tensor<BM>::API::stream_t stream = NULL) {
    LOG(INFO) << "device tensor data";
    print_tensor(tensor);
}

template<>
double tensor_mean_value<BM>(Tensor<BM>& tensor, typename Tensor<BM>::API::stream_t stream = NULL) {
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}

template<>
double tensor_mean_value_valid<BM>(Tensor<BM>& tensor,
                                   typename Tensor<BM>::API::stream_t stream = NULL) {
    Tensor<X86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}



}
}