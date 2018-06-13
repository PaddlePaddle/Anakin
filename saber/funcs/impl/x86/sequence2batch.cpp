#include "sequence2batch.h"

namespace anakin {
namespace saber {
namespace math {

template <DataType Dtype, typename LayOutType>
void CopyMatrixRowsFunctor<Dtype, LayOutType>::operator()(
                  ioTensor* src,
                  std::vector<int> index_lod, ioTensor* dst,
                  bool is_src_index) {
    int* index = index_lod.data();
    auto src_shape = src->valid_shape();
    auto dst_shape = dst->valid_shape();
    /*if (src_shape.size() != 2) {
        LOG(ERROR) << "The src must be matrix with rank 2.";
        exit(-1);
    }
    if (dst_shape.size() != 2) {
        LOG(ERROR) << "The dst must be matrix with rank 2.";
        exit(-1);
    }*/
    if (dst_shape[1] != src_shape[1]) {
        LOG(ERROR) << "The width of src and dst must be same.";
        exit(-1);
    }
    auto height = dst_shape[0];
    auto width = dst_shape[1];
    auto* src_data = src->data();
    auto* dst_data = dst->mutable_data();
//#pragma omp parallel for
    for (int i = 0; i < height; ++i) {
      if (is_src_index) {
        memcpy(dst_data + i * width, src_data + index[i] * width,
               width * sizeof(dtype));
      } else {
        memcpy(dst_data + index[i] * width, src_data + i * width,
               width * sizeof(dtype));
      }
    }
}

template class CopyMatrixRowsFunctor<AK_FLOAT, NCHW>;

template class LoDTensor2BatchFunctor<AK_FLOAT, NCHW>;
template class Batch2LoDTensorFunctor<AK_FLOAT, NCHW>;
template class ReorderInitState<AK_FLOAT, NCHW>;

}  // namespace math
}  // namespace saber
}  // namespace anakin
