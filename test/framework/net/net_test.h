/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_NET_TEST_H
#define ANAKIN_NET_TEST_H

#include <iostream>
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"
#include "net.h"
#include "worker.h"

using namespace anakin;
using ::anakin::test::Test;

using namespace anakin::graph;

/**
 * \brief Graph test is base Test class for anakin graph funciton.  
 */
class NetTest: public Test {
public:
    NetTest(){}

    void SetUp(){}

    void TearDown(){}

protected:
};

#ifdef USE_CUDA
void test_print(Tensor4dPtr<NV>& out_tensor_p) {
    Tensor4d<target_host<NV>::type> h_tensor_result;
    h_tensor_result.re_alloc(out_tensor_p->valid_shape());
    LOG(ERROR) << "result count : " << h_tensor_result.valid_shape().count();
    h_tensor_result.copy_from(*out_tensor_p);
    LOG(INFO) << "output num:" << h_tensor_result.valid_size();
    float * data = (float*)(h_tensor_result.mutable_data());
    for (int i = 0; i < h_tensor_result.valid_size(); i++) {
        LOG(INFO) << " GET OUT (" << i << ") " << data[i];
    }
}
#endif

#ifdef USE_X86_PLACE
void test_print(Tensor4dPtr<X86>& out_tensor_p) {
    LOG(ERROR) << "result count : " << out_tensor_p->valid_shape().count();
    LOG(INFO) << "output num:" << out_tensor_p->valid_size();
    float * data = (float*)(out_tensor_p->mutable_data());
    for (int i = 0; i < out_tensor_p->valid_size(); i++) {
        LOG(INFO) << " GET OUT (" << i << ") " << data[i];
    }
}
#endif

template<typename Ttype, DataType Dtype>
double tensor_average(Tensor4dPtr<Ttype>& out_tensor_p) {
    double sum = 0.0f;
#ifdef USE_CUDA
    float* h_data = new float[out_tensor_p->valid_size()];
    const float* d_data = out_tensor_p->data();
    CUDA_CHECK(cudaMemcpy(h_data, d_data, out_tensor_p->valid_size()*sizeof(float), cudaMemcpyDeviceToHost));
#else
	float* h_data = out_tensor_p->data();
#endif
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
		sum+=h_data[i];
    }
    return sum/*/out_tensor_p->valid_size()*/;
}

#ifdef USE_X86_PLACE
static int record_dev_tensorfile(const Tensor4d<X86>* dev_tensor, const char* locate) {
    Tensor<target_host<X86>::type> host_temp;
    host_temp.re_alloc(dev_tensor->valid_shape());
    host_temp.copy_from(*dev_tensor);
    const float* data = (const float*)(host_temp.data());
    FILE* fp = fopen(locate, "w+");
    int size = host_temp.valid_shape().count();
    if (fp == 0) {
        LOG(ERROR) << "[ FAILED ] file open target txt: " << locate;
    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%.18f \n", i, (data[i]));
        }
        fclose(fp);
    }
    LOG(INFO) << "[ SUCCESS ] Write " << size << " data to: " << locate;
    return 0;
}
#endif

#endif


