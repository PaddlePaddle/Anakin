/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#include <string>
#include "framework/graph/graph.h"
#include "framework/core/net/net.h"
#include "saber/core/tensor.h"
#include "saber/funcs/timer.h"


std::string g_model_path = "xxx.anakin.bin";
int g_batch_size = 1;
int g_thread_num = 4;
int g_warming_epoch = 5;
int g_run_epoch = 50;
void run_demo(){

    //init graph object, graph is the skeleton of model
    anakin::graph::Graph<anakin::saber::NV, anakin::Precision::FP32>* graph = new
            anakin::graph::Graph<anakin::saber::NV, anakin::Precision::FP32>();

    //!! load model from file to init the graph
    auto status = graph->load(g_model_path);
    std::cout << "model_path:" <<g_model_path << std::endl;

    //get inputs name in net
    std::vector<std::string>& vin_name = graph->get_ins();

    //reset batch size by max input batch
    for (auto& in: graph->get_ins()) {
        graph->ResetBatchSize(in, g_batch_size);
    }
    //!! you have to optimize the graph to get good performance
    auto status_2 = graph->Optimize();

    //!! net_executer is the executor object of model.
    anakin::Net<anakin::saber::NV, anakin::Precision::FP32> net(true);

    //!! use graph to init net, flag true means use automatic layout schedule
    net.init(*graph, true);

    //set sequence offset, which mean how to segment the words
    std::vector<std::vector<int>> seq_offset = {{0, g_batch_size}};

    for (int i = 0; i < vin_name.size(); i++) {
        //use input string to get the input tensor of net. for we use NV as target, the tensor of net_executer is on host memory
        anakin::saber::Tensor<anakin::saber::NV>* tensor = net.get_in(vin_name[i]);
        //!! fill input here
        fill_tensor_rand(*tensor);
        //!! seq_offset is for nlp model, determine the word of sequence belong which sentence, it is the same description with LOD of Paddle
        tensor->set_seq_offset(seq_offset);
    }

    //!! run net
    net.prediction();

    std::vector<std::string>& out_name = graph->get_outs();
    for (int i=0; i<out_name.size(); i++) {
        //!! get output content, we just print tensor avg here, you could copy it to next stage
        std::cout <<"net output avg = "<<tensor_mean_value_valid(*net.get_out(out_name[i]));
    }

    //performance  test(latency) using  anakin saber timer, just for demo
    anakin::saber::Context<anakin::saber::NV> ctx(0, 0, 0);
    anakin::saber::SaberTimer<anakin::saber::NV> my_timer;
    for (int i = 0; i < g_warming_epoch; i++) {
        net.prediction();
    }
    for (int i = 0; i < g_run_epoch; i++) {
        my_timer.start(ctx);
        net.prediction();
        cudaDeviceSynchronize();
        my_timer.end(ctx);
    }

    std::cout << "==========================Performance Statistics =============================\n";
    std::cout << "==================== Input_shape:       ["
              << net.get_out(out_name[0])->num() << ", "
              << net.get_out(out_name[0])->channel() << ", "
              << net.get_out(out_name[0])->height() << ", "
              << net.get_out(out_name[0])->width() << "]\n";
    std::cout << "==================== Warm_up:           " << g_warming_epoch << "\n";
    std::cout << "==================== Iteration:         " << g_run_epoch << "\n";
    std::cout << "==================== Average time:      " << my_timer.get_average_ms()  << "ms\n";
    std::cout << "==================== 10% Quantile time: " << my_timer.get_tile_time(10) << "ms\n";
    std::cout << "==================== 25% Quantile time: " << my_timer.get_tile_time(25) << "ms\n";
    std::cout << "==================== 50% Quantile time: " << my_timer.get_tile_time(50) << "ms\n";
    std::cout << "==================== 75% Quantile time: " << my_timer.get_tile_time(75) << "ms\n";
    std::cout << "==================== 90% Quantile time: " << my_timer.get_tile_time(90) << "ms\n";
    std::cout << "==================== 95% Quantile time: " << my_timer.get_tile_time(95) << "ms\n";
    std::cout << "==================== 99% Quantile time: " << my_timer.get_tile_time(99) << "ms" << std::endl;
    delete graph;
}

int main(int argc, const char** argv){
    if (argc >= 2){
        g_model_path = argv[1];
    }
    std::cout << "model_path:" << g_model_path << std::endl;
    if (argc >= 3){
        g_batch_size = atoi(argv[2]);
    }
    anakin::saber::Env<anakin::saber::NV>::env_init();
    run_demo();
    return 0;
}
