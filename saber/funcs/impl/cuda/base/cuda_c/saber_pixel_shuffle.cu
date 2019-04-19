#include "saber/funcs/impl/cuda/saber_pixel_shuffle.h"

namespace anakin{
namespace saber{


template <typename Dtype>
__global__ void ker_permute_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        int org_idx = tid;
        int in_idx = 0;
        #pragma unroll
        for (int i = 0; i < num_axes; i++) {
            int order = permute_order[i];
            int new_step = new_steps[i];
            int old_step = old_steps[order];
            in_idx += (org_idx / new_step) * old_step;
            org_idx %= new_step;
        }
        out_data[tid] = in_data[in_idx];
    }
}



template <>
SaberStatus SaberPixelShuffle<NV, AK_FLOAT>::dispatch(
	                             const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 PixelShuffleParam<NV> &param){

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    
    const float* in_data = static_cast<const float*>(inputs[0]->data());
    float* out_data = static_cast<float*>(outputs[0]->mutable_data());

    const int* permute_order = static_cast<const int*>(_permute_order.data());
    const int* new_steps = static_cast<const int*>(_out_step.data());
    const int* old_steps = static_cast<const int*>(_in_step.data());

    int count = outputs[0]->valid_size();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()){
    	ker_permute_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _axes, count, permute_order, \
                        new_steps, old_steps, in_data);
    } else {
    	ker_permute_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _axes, count, permute_order, \
                        new_steps, old_steps, in_data);
    }

}





}

}