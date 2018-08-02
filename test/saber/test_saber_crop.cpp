#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/crop.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include <vector>


using namespace anakin::saber;

template<typename TargetType, typename TargetType_H>
void test_crop(std::vector< Tensor<TargetType> *>& inputs_big, std::vector< Tensor<TargetType> *>& outputs_big,
               std::vector< Tensor<TargetType> *>& inputs, std::vector< Tensor<TargetType>*>& outputs,
               Shape input_offset, Shape output_offset,
               bool input_share, bool output_share, bool input_share_sub, bool output_share_sub,
               CropParam<TargetType>& param, bool get_time, Context<TargetType>& ctx) {
    typedef Tensor<TargetType_H> TensorH;
    typedef Tensor<TargetType> TensorD;
    /*prepare data*/
    Crop<TargetType, AK_FLOAT> crop;
    crop.compute_output_shape(inputs, outputs, param);
    outputs[0]->set_shape(outputs[0]->valid_shape(), outputs[0]->valid_shape());
    inputs[0]->set_shape(inputs[0]->valid_shape(), inputs[0]->valid_shape());

    if (output_share && output_share_sub) {
        outputs[0]->share_sub_buffer(*outputs_big[0], outputs[0]->valid_shape(), output_offset);
    } else if (output_share) {
        outputs[0]->share_from(*outputs_big[0]);
    } else {
        outputs[0]->re_alloc(outputs[0]->valid_shape(), AK_FLOAT);
    }

    if (input_share && input_share_sub) {
        inputs[0]->share_sub_buffer(*inputs_big[0], inputs[0]->valid_shape(), input_offset);
    } else if (input_share) {
        inputs[0]->share_from(*inputs_big[0]);
    } else {
#ifdef USE_CUDA
        inputs[0]->re_alloc(inputs[0]->valid_shape(), AK_FLOAT);
        cudaMemcpy(inputs[0]->mutable_data(), inputs_big[0]->data(),
                   sizeof(float) * (inputs[0]->valid_size()), cudaMemcpyDeviceToDevice);
#endif
    }

    // init assume output tensor has been reshpaed by user./
    
    crop.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);

    /*warm up*/
    crop(inputs, outputs, param,ctx);
    //crop.dispatch(inputs, outputs, param);
    
    typename TensorD::API::stream_t stream = ctx.get_compute_stream();
    outputs[0]->record_event(stream);
    outputs[0]->sync();
    Tensor<TargetType>::API::device_sync();
    /*test time
    if (get_time) {
        SaberTimer<TargetType> my_time;
        my_time.start(ctx);

        for (int i = 0; i < 100; i++) {
            crop(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        //LOG(INFO)<<"aveage time"<<my_time.get_average_ms()/100;
    }*/

   // CUDA_CHECK(cudaPeekAtLastError());
    Shape valid_shape = outputs[0]->valid_shape();
    //printf("shape: %d, %d, %d, %d\n", valid_shape[0], valid_shape[1],valid_shape[2], valid_shape[3]);
    TensorD out_valid(outputs[0]->valid_shape());
    //CUDA_CHECK(cudaPeekAtLastError());
    out_valid.copy_from(*outputs[0]);
    //CUDA_CHECK(cudaPeekAtLastError());

    print_tensor(*inputs[0]);
    //cudaDeviceSynchronize();
    //print_tensor_device(*outputs[0]);
    //cudaDeviceSynchronize();
    print_tensor(out_valid);
   // cudaDeviceSynchronize();
   // CUDA_CHECK(cudaPeekAtLastError());
   
}

TEST(TestSaberFunc, test_func_constructor) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    typedef Tensor<NVHX86> TensorH;
    typedef Tensor<NV> TensorD;
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    typedef Tensor<X86> TensorH;
    typedef Tensor<X86> TensorD;
#endif


    int n = 1;
    int c = 5;
    int h = 8;
    int w = 8;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape real_shape({n + 1, c, h + 1, w + 1},Layout_NCHW);
    Shape valid_shape({n, c, h, w},Layout_NCHW);
    Shape input_offset({0, 0, 0, 0},Layout_NCHW);
    Shape output_offset({0, 0, 0, 0},Layout_NCHW);

    TensorH in_host_big;
    TensorD in_dev_big;
    TensorH out_host_big;
    TensorD out_dev_big;

    TensorD in_dev;
    TensorD output_dev;

    in_host_big.re_alloc(real_shape, AK_FLOAT);
    in_dev_big.re_alloc(real_shape, AK_FLOAT);
    out_host_big.re_alloc(real_shape, AK_FLOAT);
    out_dev_big.re_alloc(real_shape, AK_FLOAT);

    /*prepare input data*/

 
    float* indata=(float*)in_host_big.mutable_data();
    fill_tensor_rand(in_host_big, -1, 1);
    for(int i=0;i<in_host_big.size();++i){
    		indata[i]=i;
    }

    in_dev_big.copy_from(in_host_big);


    in_dev.set_shape(valid_shape);

    std::vector<TensorD*> inputs_big;
    std::vector<TensorD*> outputs_big;
    std::vector<TensorD*> inputs;
    std::vector<TensorD*> outputs;
    inputs_big.push_back(&in_dev_big);
    outputs_big.push_back(&out_dev_big);
    inputs.push_back(&in_dev);
    outputs.push_back(&output_dev);
#ifdef USE_CUDA
    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    std::vector<int> offset = {2, 2};
    std::vector<int> shape = {1, 3, 4, 5};
    CropParam<NV> param(/* axis_in*/ 2, /*offset*/offset, /*shape_in*/shape);

#endif
#ifdef USE_X86_PLACE
    // start Reshape & doInfer
    Context<X86> ctx(0, 1, 1);
    std::vector<int> offset = {2, 2};
    std::vector<int> shape = {1, 3, 4, 5};
    CropParam<X86> param(/* axis_in*/ 2, /*offset*/offset, /*shape_in*/shape);

#endif
  
    for (auto input_share : {
                false
            }) {
        for (auto output_share : {
                    false
                }) {
            for (auto input_share_sub : {
                        false
                    }) {
                for (auto output_share_sub : {
                            false
                        }) {
                    for (auto get_time : {
                                false
                            }) {
                        LOG(INFO) << input_share << "," << output_share << "," << input_share_sub << "," << output_share_sub
                                << "," << get_time;
                                
#ifdef USE_CUDA
                                test_crop<NV,NVHX86>(inputs_big, outputs_big,inputs, outputs,input_offset, output_offset,input_share, output_share, input_share_sub,output_share_sub,param, get_time, ctx);
#endif
#ifdef USE_X86_PLACE
                                test_crop<X86,X86>(inputs_big, outputs_big,
                                                   inputs, outputs,
                                                   input_offset, output_offset,
                                                   true, output_share, true, output_share_sub,
                                                   param, get_time, ctx);
#endif
                     
                    }
                }
            }
        }
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

