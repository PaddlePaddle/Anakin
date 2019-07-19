#include "saber/funcs/impl/x86/sequence2batch.h"

namespace anakin {
namespace saber {
namespace math {

template <DataType Dtype, typename LayOutType>
void CopyMatrixRowsFunctor<Dtype, LayOutType>::operator()(
    ioTensor* src,
    std::vector<int> index_lod, ioTensor* dst,
    bool is_src_index, int fragment_num, int offset, int width) {
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
    }
    if (dst_shape[1] != src_shape[1]) {
        LOG(ERROR) << "The width of src and dst must be same.";
        exit(-1);
    }*/
    if (dst_shape[1] % fragment_num != 0 && src_shape[1] % fragment_num != 0) {
        LOG(ERROR) << "hidden size should be divided with no remainder by fragment_num.";
        exit(-1);
    }
    typedef typename DataTrait<X86,Dtype>::PtrDtype Data_ptr;

    auto height = dst_shape[0];
    auto dst_width = dst_shape[1] / fragment_num;
    auto src_width = src_shape[1] / fragment_num;
    auto real_width = (width != 0) ? width : (dst_width > src_width ? src_width : dst_width);
    Data_ptr src_data = static_cast<Data_ptr>(src->data());
    Data_ptr dst_data = static_cast<Data_ptr>(dst->mutable_data());

    if (is_src_index) {
        #pragma omp parallel for collapse(2)

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < fragment_num; j++) {
                memcpy(dst_data + i * fragment_num * dst_width + j * dst_width + offset,
                       src_data + index[i] * fragment_num * src_width + j * src_width,
                       real_width * sizeof(dtype));
            }
        }
    } else {
        #pragma omp parallel for collapse(2)

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < fragment_num; j++) {
                memcpy(dst_data + index[i] * fragment_num * dst_width + j * dst_width + offset,
                       src_data + i * fragment_num * src_width + j * src_width,
                       real_width * sizeof(dtype));
            }
        }
    }
}

template class CopyMatrixRowsFunctor<AK_FLOAT, NCHW>;

template class Seq2BatchFunctor<AK_FLOAT, NCHW>;
template class Batch2SeqFunctor<AK_FLOAT, NCHW>;
template class ReorderInitState<AK_FLOAT, NCHW>;

}  // namespace math
}  // namespace saber
}  // namespace anakin
