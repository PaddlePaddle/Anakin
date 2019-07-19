#include "saber/core/context.h"
#include "saber/funcs/reduce_min.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype>
void reduce_n(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    
    int hw_size = height_in * width_in;
    int chw_size = channel_in * hw_size;
    int data_index = 0;
    int src_index = 0;
    int src_index0 = 0;
    for (int c = 0; c < channel_in; ++c) {
        for (int h = 0; h < height_in; ++h) {
            for (int w = 0; w < width_in; ++w) {
                data_index = c * hw_size + h * width_in + w;
                dst[data_index] = src[data_index];
                for (int n = 1; n < num_in; ++n) {
                    src_index = n * chw_size + data_index;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_c(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int hw_size = height_in * width_in;
    int chw_size = hw_size * channel_in;  
    int data_index = 0;
    int src_index0 = 0;
    int src_index = 0;
    for (int n = 0; n < num_in; ++n) {
        for (int h = 0; h < height_in; ++h) {
            for (int w = 0; w < width_in; ++w) {
                data_index = n * hw_size + h * width_in + w;
                src_index0 = n * chw_size + h * width_in + w; 
                dst[data_index] = src[src_index0];
                for (int c = 1; c < channel_in; ++c) {
                    src_index = src_index0 + c * hw_size;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_h(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int cw_size = channel_in * width_in;
    int chw_size = cw_size * height_in;
    int hw_size = height_in * width_in;
    int data_index = 0;
    int src_index = 0;
    int src_index0 = 0;
    for (int n = 0; n < num_in; ++n) {
        for (int c = 0; c < channel_in; ++c) {
            for (int w = 0; w < width_in; ++w) {
                data_index = n * cw_size + c * width_in + w;
                src_index0 = n * chw_size + c * hw_size + w;
                dst[data_index] = src[src_index0];
                for (int h = 1; h < height_in; ++h) {
                    src_index = src_index0 + h * width_in;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_w(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int ch_size = channel_in * height_in;
    int hw_size = height_in * width_in;
    int chw_size = ch_size * width_in;
    int data_index = 0; 
    int src_index0 = 0;
    int src_index = 0;
    for (int n = 0; n < num_in; ++n) {
        for (int c = 0; c < channel_in; ++c) {
            for (int h = 0; h < height_in; ++h) {
                data_index = n * ch_size + c * height_in + h;
                src_index0 = n * chw_size + c * hw_size + h * width_in;
                dst[data_index] = src[src_index0];
                for (int w = 1; w < width_in; ++w) {
                    src_index = src_index0 + w;
                    dst[data_index] = dst[data_index] < src[src_index] ? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_all(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    dtype min = src[0];
    int src_index = 0;
    int n_id = 0;
    int c_id = 0;
    for (int n = 0; n < num_in; ++n) {
        n_id = n * channel_in * height_in * width_in;
        for (int c = 0; c < channel_in; ++c) {
            c_id = c * height_in * width_in;
            for (int h = 0; h < height_in; ++h) {
                for (int w = 0; w < width_in; ++w) {
                    src_index = n_id + c_id + h * width_in + w;
                    min = src[src_index] < min? src[src_index] : min;
                }
            }
        }
    }
    dst[0] = min;
}
template <typename dtype, typename TargetType_H>
void reduce_nc(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    
    //reduce n first. 
    Shape shape_tmp({1, channel_in, height_in, width_in});
    Tensor<TargetType_H> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_n<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_c<dtype>(tmp_out, dst, 1, channel_in, height_in, width_in);
}

template <typename dtype, typename TargetType_H>
void reduce_ch(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    //reduce c first
    Shape shape_tmp({num_in, 1, height_in, width_in});
    Tensor<TargetType_H> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_c<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_h<dtype>(tmp_out, dst, num_in, 1, height_in, width_in); 
}

template <typename dtype, typename TargetType_H>
void reduce_hw(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    //reduce h first
    Shape shape_tmp({num_in, channel_in, 1, width_in});
    Tensor<TargetType_H> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_h<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_w<dtype>(tmp_out, dst, num_in, channel_in, 1, width_in); 
}

/**
 * @brief This operator is to reduce input tensor according to the given dimentions.
 *            For details, please see saber_reduce_min.cu.
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input 
 * @param output 
 * @param param 
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void reduce_min_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, ReduceMinParam<TargetType_D>& param) {
    
    int n = input[0]->num();
    int c = input[0]->channel();
    int h = input[0]->height();
    int w = input[0]->width();
    int count = input[0]->valid_size();
    int rank = input[0]->valid_shape().size();
    const dtype* input_ptr = (const dtype*)input[0]->data();
    dtype* output_ptr = (dtype*)output[0]->mutable_data();
    std::vector<int> reduce_dim = param.reduce_dim;
    //we don't need to check whether reduce_dim is valid because it will be checked in cuda/x86 impl.
    if (!reduce_dim.empty()) {
        //not empty
        for (int i = 0; i < reduce_dim.size(); ++i) {
            if (reduce_dim[i] < 0) {
                reduce_dim[i] += rank;
            }
        }
    }

    if (reduce_dim.empty()) {
        //reduce all.
        reduce_all<dtype>(input_ptr, output_ptr, n, c, h, w);
    }else {
        if (reduce_dim.size() == 1) {
            switch (reduce_dim[0]) {
                case 0: reduce_n<dtype>(input_ptr, output_ptr, n, c, h, w); break;
                case 1: reduce_c<dtype>(input_ptr, output_ptr, n, c, h, w); break;
                case 2: reduce_h<dtype>(input_ptr, output_ptr, n, c, h, w); break;
                case 3: reduce_w<dtype>(input_ptr, output_ptr, n, c, h, w); break;
                default: LOG(FATAL) << "error!!!";
            }
        }else if (reduce_dim.size() == 2) {
            if (reduce_dim[0] == 0 && reduce_dim[1] == 1) {
                reduce_nc<dtype, TargetType_H>(input_ptr, output_ptr, n, c, h, w);
            }else if (reduce_dim[0] == 1 && reduce_dim[1] == 2) {
                reduce_ch<dtype, TargetType_H>(input_ptr, output_ptr, n, c, h, w);
            }else if (reduce_dim[0] == 2 && reduce_dim[1] == 3) {
                reduce_hw<dtype, TargetType_H>(input_ptr, output_ptr, n, c, h, w);
            }else {
                LOG(FATAL) <<"invalid reduce_dim!!";
            }
        } else {
            LOG(FATAL) << "reduce_dim's size over than 2, which is not supported now!!";
        }
    }

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_reduce_min(){
    TestSaberBase<TargetType_D, TargetType_H, Dtype, ReduceMin, ReduceMinParam> testbase;
    std::vector<int> reduce_dim{2, 3};
    ReduceMinParam<TargetType_D> param(reduce_dim, false);

    for (int w_in : {2, 8, 16, 32}) {
        for (int h_in : {2, 8, 16, 32, 64}) {
            for (int ch_in : {2, 7, 8, 64}) {
                for (int num_in:{2, 21, 32, 64}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase.set_param(param);
                    //testbase.set_rand_limit();
                    testbase.set_input_shape(shape);
                    testbase.run_test(reduce_min_cpu_base<float, TargetType_D, TargetType_H>);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_ReduceMin) {

#ifdef USE_CUDA
   //Init the test_base
    test_reduce_min<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_reduce_min<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_ReduceMin<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
