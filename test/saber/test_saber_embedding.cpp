#include "saber/core/context.h"
#include "saber/funcs/embedding.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include <vector>
#include <ctime>

using namespace anakin::saber;

template<typename TargetType>
void embedding_cpu(int input_num, int channels, int height, int width,  
                   int word_num, int emb_dim, int padding_idx,
                   Tensor<TargetType> &input, Tensor<TargetType> &output, 
                   const Tensor<TargetType> &weight) {

	const float *in_data = (const float*)input.data();
	int num = input.valid_size();
    float *out_data = (float*)output.mutable_data();
    for (int i = 0; i < num; i++) {
        if (in_data[i] == padding_idx) {
            memset(out_data + i * emb_dim, 0, sizeof(float) * emb_dim);
        } else {
            CHECK_GE(int(in_data[i]), 0);
            CHECK_LT(int(in_data[i]), word_num);
            memcpy(out_data + i * emb_dim, (const float*)weight.data()+int(in_data[i]) * emb_dim, sizeof(float) * emb_dim);  
        }
    }
}

//Produce int data
template<typename TargetType>
void fill_tensor_rand_int(Tensor<TargetType> &tensor, int st, int ed) {
	srand((unsigned)time(NULL));
	float *data = (float*)tensor.mutable_data();
	for (int i = 0; i < tensor.size(); i++) {
		data[i] = rand() % (ed - st) + st;
	}
}
template<typename TargetType, typename TargetType_H>
void test_embedding_results(int input_num, int channels, int height, int width, 
                            int word_num, int emb_dim, int padding_idx) {
    
    LOG(INFO) << " Embedding Params: ";
    LOG(INFO) << " input_num: " << input_num;
    LOG(INFO) << " channels: " << channels;
    LOG(INFO) << " height: " << height;
    LOG(INFO) << " width: " << width;
    LOG(INFO) << " word_num: " << word_num;
    LOG(INFO) << " emb_dim: " << emb_dim;
    LOG(INFO) << " padding_idx: " << padding_idx;
    
    Shape input_s({input_num, channels, height, width}, Layout_NCHW);
    Shape weights_s({1, 1, word_num, emb_dim}, Layout_NCHW);
	Shape output_s({1, 1, input_num, emb_dim}, Layout_NCHW);

    //Init input tensor
    Tensor<TargetType> input_d;
    Tensor<TargetType_H> input_h;
    Tensor<TargetType_H> input_h_temp;
    input_d.re_alloc(input_s, AK_FLOAT);
    input_h.re_alloc(input_s, AK_FLOAT);
    input_h_temp.re_alloc(input_s, AK_FLOAT);

    fill_tensor_rand_int(input_h_temp, 0, 128);
//	print_tensor(input_h_temp);
    input_h.copy_from(input_h_temp);

    input_d.copy_from(input_h);


    //Weight tensor
    Tensor<TargetType> weight_d;
    Tensor<TargetType_H> weight_h;
	Tensor<TargetType_H> weight_h_temp;
    weight_h.re_alloc(weights_s, AK_FLOAT);
    weight_d.re_alloc(weights_s, AK_FLOAT);
    weight_h_temp.re_alloc(weights_s, AK_FLOAT); 
	fill_tensor_rand(weight_h, -0.5, 1.5);
    weight_d.copy_from(weight_h);
	weight_h_temp.copy_from(weight_h);

    //Output tensor
    Tensor<TargetType> output_d;
    Tensor<TargetType_H> output_d_h;
    Tensor<TargetType_H> output_h;

    //Host computation
    output_h.re_alloc(output_s, AK_FLOAT);
    embedding_cpu(input_num, channels, height, width, 
                  word_num, emb_dim, padding_idx, input_h_temp, output_h, weight_h_temp);


    //Device computation
    Context<TargetType> ctx(0, 1, 1);
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_d);
    output_v.push_back(&output_d);

    EmbeddingParam<TargetType> param(word_num, emb_dim, padding_idx, &weight_d);
    Embedding<TargetType, AK_FLOAT> emb;
    emb.compute_output_shape(input_v, output_v, param);
    output_d.re_alloc(output_d.valid_shape(), AK_FLOAT);

    emb.init(input_v, output_v, param, SPECIFY, SABER_IMPL, ctx);
    emb(input_v, output_v, param, ctx);

    typename Tensor<TargetType>::API::stream_t stream = ctx.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_d_h.re_alloc(output_d.valid_shape(), AK_FLOAT);
    output_d_h.copy_from(output_d);

    double max_ratio = 0.0;
    double max_diff = 0.0;

    tensor_cmp_host((const float*)output_d_h.data(), (const float*)output_h.data(), 
                     output_h.valid_size(), max_ratio, max_diff);

    if (max_ratio < 1e-6f) {
        LOG(INFO) << " Test passed!!! ";
    } else {
        LOG(INFO) << "Test failed!!!  max_ratio = "<< max_ratio << " max_diff = "<<max_diff;
    }

}

template<typename TargetType, typename TargetType_H>
void test_embedding_performance(int input_num, int channels, int height, int width, 
                            int word_num, int emb_dim, int padding_idx) { 
    LOG(INFO) << " Embedding Params: ";
    LOG(INFO) << " input_num: " << input_num;
    LOG(INFO) << " channels: " << channels;
    LOG(INFO) << " height: " << height;
    LOG(INFO) << " width: " << width;
    LOG(INFO) << " word_num: " << word_num;
    LOG(INFO) << " emb_dim: " << emb_dim;
    LOG(INFO) << " padding_idx: " << padding_idx;
    
    Shape input_s({input_num, channels, height, width}, Layout_NCHW);
    Shape weights_s({1, 1, word_num, emb_dim}, Layout_NCHW);
	Shape output_s({1, 1, input_num, emb_dim}, Layout_NCHW);
    int iter = 1000;
    float elapsedTime;

    //Init input tensor
    Tensor<TargetType> input_d;
    Tensor<TargetType_H> input_h;
    Tensor<TargetType_H> input_h_temp;
    input_d.re_alloc(input_s, AK_FLOAT);
    input_h.re_alloc(input_s, AK_FLOAT);
    input_h_temp.re_alloc(input_s, AK_FLOAT);

    fill_tensor_rand_int(input_h_temp, 0, 128);
    input_h.copy_from(input_h_temp);
    input_d.copy_from(input_h);

    //Weight tensor
    Tensor<TargetType> weight_d;
    Tensor<TargetType_H> weight_h;
	Tensor<TargetType_H> weight_h_temp;
    weight_h.re_alloc(weights_s, AK_FLOAT);
    weight_d.re_alloc(weights_s, AK_FLOAT);
    weight_h_temp.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weight_h, -0.5, 0.5);
    weight_d.copy_from(weight_h);
	weight_h_temp.copy_from(weight_h);

    //Output tensor
    Tensor<TargetType> output_d;
    Tensor<TargetType_H> output_d_h;
    Tensor<TargetType_H> output_h;

    Context<TargetType_H> ctx_host;
    SaberTimer<TargetType_H> t;
    t.clear();
    t.start(ctx_host);

    output_h.re_alloc(output_s, AK_FLOAT);
    for (int i = 0; i < 1; i++) {
            embedding_cpu(input_num, channels, height, width, 
                  word_num, emb_dim, padding_idx, input_h_temp, output_h, weight_h_temp);
    }

    t.end(ctx_host);
    elapsedTime = t.get_average_ms();
    printf("CPU total time for running 1000 times: %.4f ms, avg time : %.4f ms\n", elapsedTime, elapsedTime / iter);


    //Device computation
    Context<TargetType> ctx(0, 1, 1);
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_d);
    output_v.push_back(&output_d);

    EmbeddingParam<TargetType> param(word_num, emb_dim, padding_idx, &weight_d);
    Embedding<TargetType, AK_FLOAT> emb;
    emb.compute_output_shape(input_v, output_v, param);
    output_d.re_alloc(output_d.valid_shape(), AK_FLOAT);

    emb.init(input_v, output_v, param, SPECIFY, SABER_IMPL, ctx);

    SaberTimer<TargetType> t1;
    t1.clear();
    t1.start(ctx);

    for ( int i = 0; i < 1 ; i++) {
        emb(input_v, output_v, param, ctx);
        output_v[0]->record_event(ctx.get_compute_stream());
        output_v[0]->sync();
    }

    t1.end(ctx);
    elapsedTime = t1.get_average_ms();
    printf("GPU total time for running 1000 times: %.4f ms, avg time : %.4f ms\n", elapsedTime, elapsedTime / iter);

}


TEST(TestSaberFunc, test_embedding_results_t) {
    int input_num = 103;
    int channels = 1;
    int height = 1;
    int width = 1;
    int word_num = 128;
    int emb_dim = 10;
    int padding_idx = -1;

    test_embedding_results<NV, NVHX86>(input_num, channels, height, width, 
                                       word_num, emb_dim, padding_idx);


}

TEST(TestSaberFunc, test_embedding_performance_t) {
    int input_num = 100;
    int channels = 1;
    int height = 1;
    int width = 1;
    int word_num = 128;
    int emb_dim = 10;
    int padding_idx = -1;

    test_embedding_performance<NV, NVHX86>(input_num, channels, height, width, 
                                       word_num, emb_dim, padding_idx);

}

TEST(TestSaberFunc, test_embedding_func) {

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
	Env<NV>::env_init();
    Env<NVHX86>::env_init();InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
