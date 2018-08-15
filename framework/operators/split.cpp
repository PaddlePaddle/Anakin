#include "framework/operators/split.h"

namespace anakin {

namespace ops {
#define INSTANCE_SPLIT(Ttype, Dtype, Ptype) \
template<> \
void Split<Ttype, Dtype, Ptype>::operator()( \
        OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {}


template<typename Ttype, DataType Dtype, Precision Ptype>
Status SplitHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Split op parameter.";
    split_num = GET_PARAMETER(int, split_num);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SplitHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                         std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SplitHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                               std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    for (int i = 0; i < split_num; i++) {
        outs[i]->set_shape(ins[0]->valid_shape());
        outs[i]->set_seq_offset(ins[0]->get_seq_offset());
    }
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_SPLIT(NV, AK_FLOAT, Precision::FP32);
template class SplitHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SPLIT(ARM, AK_FLOAT, Precision::FP32);
template class SplitHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SPLIT(X86, AK_FLOAT, Precision::FP32);
template class SplitHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Split)
.Doc("Split operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("split")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("split")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("split")
#endif
.num_in(1)
.num_out(1)
.Args<int>("split_num", " split output number. ");


} /* namespace ops */

} /* namespace anakin */


