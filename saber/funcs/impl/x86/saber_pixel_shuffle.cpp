#include "saber/funcs/impl/x86/saber_pixel_shuffle.h"

namespace anakin{
namespace saber{
template class SaberPixelShuffle<X86, AK_FLOAT>;

template <>
SaberStatus SaberPixelShuffle<X86, AK_FLOAT>::\
dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 PixelShuffleParam<X86> &param){

        const float* src_ptr = static_cast<const float*>(inputs[0]->data());
        float* dst_ptr = static_cast<float*>(outputs[0]->mutable_data());
        
        int out_size = outputs[0]->valid_size();

        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()){
            for (int j = 0; j < out_size; ++j){
                int in_idx = 0;
                int id = j;
                for (int i = 0; i < _num_axes; ++i) {
                    int order = _order[i];
                    int new_step = _out_steps[i];
                    int old_step = _in_steps[order];
                    int offset = (id / new_step) * old_step;
                    in_idx += offset;
                    id %= new_step;
                }
                dst_ptr[j] = src_ptr[in_idx];
            }
        } else {
            for (int j=0; j<out_size; ++j){
                int in_idx = 0;
                int out_idx  = 0;
                int new_valid_stride = 1;
                for (int i = _num_axes - 1; i >= 0; --i) {
                    int order = _order[i];
                    int new_step = _out_steps[i];
                    int old_step = _in_steps[order];
                    int id = (j / new_valid_stride) % _out_new_sh[i];
                    in_idx += id * old_step;
                    out_idx += id * new_step;
                    new_valid_stride *= _out_new_sh[i];
                }
                dst_ptr[out_idx] = src_ptr[in_idx];
            }
        }
        return SaberSuccess;
}


}
}
