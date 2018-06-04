/* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 
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
#include "test_saber_func_AMD.h"
#include "saber/saber_funcs_param.h"
#include "core/context.h"
#include "funcs/fc.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>

using namespace anakin::saber;

typedef TargetWrapper<AMD> API;
typedef Tensor<AMD, AK_FLOAT, NCHW> TensorDf4;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef TensorDf4::Dtype ftype;

typedef struct ProblemConfigType
{
    std::string ConfigName;
    int M;
    int N, K;
    int num, channel, height, width;
    int axis;
    bool is_transpose_weight;
}T_ProblemConfig;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor &t0) {

            LOG(INFO) << name << " valid shape is ["
                      << t0.valid_shape()[0] << ", "
                      << t0.valid_shape()[1] << ", "
                      << t0.valid_shape()[2] << ", "
                      << t0.valid_shape()[3] << "].";

            LOG(INFO) << name << " real shape is ["
                      << t0.shape()[0] << ", "
                      << t0.shape()[1] << ", "
                      << t0.shape()[2] << ", "
                      << t0.shape()[3] << "].";

            LOG(INFO) << name << " offset is ["
                      << t0.offset()[0] << ", "
                      << t0.offset()[1] << ", "
                      << t0.offset()[2] << ", "
                      << t0.offset()[3] << "].";
}

#define MIOPEN_RETURNS(...) \
    ->decltype(__VA_ARGS__) { return __VA_ARGS__; }

template <class R>
using range_value = typename std::decay<decltype(*std::declval<R>().begin())>::type;

struct sum_fn
{
	template <class T, class U>
	auto operator()(T x, U y) const MIOPEN_RETURNS(x + y);
};
static constexpr sum_fn sum{};

struct compare_mag_fn
{
	template <class T, class U>
	bool operator()(T x, U y) const
	{
		return std::fabs(x) < std::fabs(y);
	}
};
static constexpr compare_mag_fn compare_mag{};

struct square_diff_fn
{
	template <class T, class U>
	double operator()(T x, U y) const
	{
		return (x - y) * (x - y);
	}
};
static constexpr square_diff_fn square_diff{};

template <class R1>
auto range_distance(R1&& r1) MIOPEN_RETURNS(std::distance(r1.begin(), r1.end()));

template <class R1, class R2, class T, class Reducer, class Product>
T range_product(R1&& r1, R2&& r2, T state, Reducer r, Product p)
{
	return std::inner_product(r1.begin(), r1.end(), r2.begin(), state, r, p);
}

template <class R1, class R2>
double rms_range(R1&& r1, R2&& r2)
{
	std::size_t n = range_distance(r1);
	if (n == range_distance(r2))
	{
		double square_difference = range_product(r1, r2, 0.0, sum_fn{}, square_diff);
		double mag1 = *std::max_element(r1.begin(), r1.end(), compare_mag);
		double mag2 = *std::max_element(r2.begin(), r2.end(), compare_mag);
		double mag =
			std::max({ std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min() });
		printf("square_difference:%f, mag1:%f, mag2:%f, mag:%f\n");
		return std::sqrt(square_difference) / (std::sqrt(n) * mag);
	}
	else
		return std::numeric_limits<range_value<R1>>::max();
}
static void Verify(TensorHf4& h_out, TensorHf4& out_ref, int size)
{
    LOG(INFO) << "Verify";

    std::vector<float> host_out;
    std::vector<float> device_out;
    for (int i = 0; i < size; i++)
    {
        host_out.push_back(h_out.data()[i]);
        device_out.push_back(out_ref.data()[i]);
        //LOG(INFO) << "I:" << i << ",host_value:" << h_out.data()[i] << ",device_value:" << out_ref.data()[i] << ",error:" << fabs(out_ref.data()[i] - h_out.data()[i]);
    }
    auto diff = rms_range(host_out, device_out);

    LOG(INFO) << "mean err: " << diff;
    if (diff > 1e-6)
    {
        LOG(INFO) << "err: " << diff;
        LOG(INFO) << "verify failed!";
    } else {
        LOG(INFO) << "verify success.";
    }
}

static void fc_host(TensorHf4& out_host, const TensorHf4& in_host,
                    const TensorHf4& weight_host, const TensorHf4& bias_host,
                    bool is_transpose_weight)
{
    printf("fc host.\n");

    int M = in_host.num();
    int K = in_host.valid_size() / M;
    int N = weight_host.valid_size() / K;
    bool bias_enable = bias_host.valid_size() > 0;

    LOG(INFO) << "M:" << M << ",K:" << K << ",N:" << N << ",bias:" << bias_enable << ",transpose:" << is_transpose_weight;

    float tmp;
    for (int y = 0; y < M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            tmp = (bias_enable == true) ? bias_host.data()[x] : 0.0f;
            for (int k = 0; k < K; k++)
            {
                int idx = (is_transpose_weight == false) ? (K * x + k) : (N * k + x);
                tmp += in_host.data()[K*y + k] * weight_host.data()[idx];
            }
            out_host.mutable_data()[N*y + x] = tmp;
        }
    }
/*
    for(int y = 0; y < M; y++)
    {
        for(int x = 0; x < N; x++)
        {
            out_host.mutable_data()[y*N + x] = 0;
            for(int k =0; k< K; k++)
            {
                out_host.mutable_data()[y*N + x] += in_host.data()[y*K + k] * weight_host.data()[K*x + k];
            }
            out_host.mutable_data()[y*N + x] += bias_host.data()[y*N + x];
        }
    }
*/
}

void test_inner_product(std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                        TensorDf4& weight, TensorDf4& bias, int num_out, int axis,
                        int is_transpose_weight, anakin::saber::ImplEnum impl, int warm_iter, int iter, Context<AMD> &ctx1, SaberTimer<AMD> &timer) {

    int noused = 0;
    FcParam<TensorDf4> param(&weight, &bias, num_out, axis, is_transpose_weight);

    //Fc<AMD, AK_FLOAT> fc;
    Fc<AMD, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> fc;
    SABER_CHECK(fc.compute_output_shape(inputs, outputs, param));

    outputs[0]->re_alloc(outputs[0]->valid_shape());

    SABER_CHECK(fc.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    for(int i = 0 ; i < warm_iter; i++) {
        fc(inputs, outputs, param, ctx1);
    }
    clFinish(ctx1.get_compute_stream());

    for(auto i = 0; i < iter; i++) {
        timer.start(ctx1);
        fc(inputs, outputs, param, ctx1);
        timer.end(ctx1);
    }
    clFlush(ctx1.get_compute_stream());
    clFinish(ctx1.get_compute_stream());

}

TEST(TestSaberFuncAMD, test_saber_fc) {

    std::list<T_ProblemConfig*> problemConfigList;
    T_ProblemConfig * problemConfig;

    TensorDf4 in;
    TensorHf4 in_host;
    TensorDf4 bias;
    TensorHf4 bias_host;
    TensorDf4 weight;
    TensorHf4 weight_host;
    TensorDf4 out;
    TensorHf4 out_host;        //to store the result from device
    TensorHf4 out_golden_host; //to store the result from cpu calculation

    for (int batch_size = 1; batch_size <= 32; batch_size *= 2) {
        if (batch_size == 16) continue;
    #if 1
        // ======================================================================
        // problem config InnerProduct NT [1,25088]*[25088,4096]:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "InnerProduct NT [1,25088]*[25088,4096]";
        problemConfig->M = batch_size;
        problemConfig->N = 4096;
        problemConfig->channel = 512;
        problemConfig->height = 7;
        problemConfig->width = 7;
        problemConfig->K = 512 * 7 * 7;
        problemConfig->axis = 1;
        problemConfig->is_transpose_weight = false;
        problemConfigList.push_back(problemConfig);

        // ======================================================================
        // problem config InnerProduct T [1,25088]*[25088,4096]:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "InnerProduct T [1,25088]*[25088,4096]";
        problemConfig->M = batch_size;
        problemConfig->N = 4096;
        problemConfig->channel = 512;
        problemConfig->height = 7;
        problemConfig->width = 7;
        problemConfig->K = 512 * 7 * 7;
        problemConfig->axis = 1;
        problemConfig->is_transpose_weight = true;
        //problemConfigList.push_back(problemConfig);
    #endif
    #if 1
        // ======================================================================
        // problem config InnerProduct NT [1,4096]*[4096,4096]:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "InnerProduct NT [1,4096]*[4096,4096]";
        problemConfig->M = batch_size;
        problemConfig->N = 4096;
        problemConfig->channel = 4096;
        problemConfig->height = 1;
        problemConfig->width = 1;
        problemConfig->K = 4096 * 1 * 1;
        problemConfig->axis = 1;
        problemConfig->is_transpose_weight = false;
        problemConfigList.push_back(problemConfig);


        // ======================================================================
        // problem config InnerProduct T [1,4096]*[4096,4096]:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "InnerProduct T [1,4096]*[4096,4096]";
        problemConfig->M = batch_size;
        problemConfig->N = 4096;
        problemConfig->channel = 4096;
        problemConfig->height = 1;
        problemConfig->width = 1;
        problemConfig->K = 4096 * 1 * 1;
        problemConfig->axis = 1;
        problemConfig->is_transpose_weight = true;
        //problemConfigList.push_back(problemConfig);
    #endif
    #if 1
        // ======================================================================
        // problem config InnerProduct [1,4096]*[4096,5]:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "InnerProduct [1,4096]*[4096,1000]";
        problemConfig->M = batch_size;
        problemConfig->N = 1000;
        problemConfig->channel = 4096;
        problemConfig->height = 1;
        problemConfig->width = 1;
        problemConfig->K = 4096 * 1 * 1;
        problemConfig->axis = 1;
        problemConfig->is_transpose_weight = false;
        problemConfigList.push_back(problemConfig);
    #endif
    }
    Context<AMD> ctx1(0, 1, 1);
    API::stream_t amd_cstream = ctx1.get_compute_stream();

    LOG(INFO) << "Total " << problemConfigList.size() << " problems...";

    SaberTimer<AMD> t_device;
    SaberTimer<AMD> t_host;

    //Begin loop
    for (auto p : problemConfigList)
    {
        //auto p = *problemConfigList.begin();
        //TODO: get the problem and solve it...
        LOG(INFO) << "Problem: " << p->ConfigName;
        
        //allocate input buffer
        LOG(INFO) << "allocate input buffer";
        in.re_alloc({p->M, p->channel, p->height, p->width});
        in_host.re_alloc({p->M, p->channel, p->height, p->width});

        //allocate weight buffer
        LOG(INFO) << "allocate weight buffer";
        if(p->is_transpose_weight) {
            weight.re_alloc({1, 1, p->K, p->N});
            weight_host.re_alloc({1, 1, p->K, p->N});
        } else {
            weight.re_alloc({1, 1, p->N, p->K});
            weight_host.re_alloc({1, 1, p->N, p->K});
        }

        //print_tensor_shape("in", in);
        //print_tensor_shape("weight", weight);

        //allocate bias buffer
        LOG(INFO) << "allocate bias buffer";
        bias.re_alloc({1, p->N, 1, 1});
        bias_host.re_alloc({1, p->N, 1, 1});

        //assign default value to input buffer
        LOG(INFO) << "assign input, bias and weight data";

        fill_tensor_device_rand(weight, 0.f, 1.f);
        fill_tensor_device_rand(bias, 0.f, 1.f);
        fill_tensor_device_rand(in, 0.f, 1.f);
        //fill_tensor_device_const(weight, 2.f);
        //fill_tensor_device_const(bias, 0.f);
        //fill_tensor_device_const(in, 1.f);
        weight_host.copy_from(weight);
        bias_host.copy_from(bias);
        in_host.copy_from(in);

        LOG(INFO) << "Weight valid_size: " << weight.valid_size();
        LOG(INFO) << "bias valid_size: " << bias.valid_size();
        LOG(INFO) << "in valid_size: " << in.valid_size();

        std::vector<TensorDf4*> input_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        test_inner_product(input_v, output_v, weight, bias, p->N, p->axis, p->is_transpose_weight, VENDER_IMPL, warm_iter, iter, ctx1, t_device);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        out_host.re_alloc(out.shape());
        out_host.copy_from(out);

        out_golden_host.re_alloc(out.shape());

        //test result in host side.
        t_host.start(ctx1);

        fc_host(out_golden_host, in_host, weight_host, bias_host, p->is_transpose_weight);
        t_host.end(ctx1);

        //LOG(INFO) << "PRINT DEVICE TENSOR: in";
        //print_tensor_device(in);
        //LOG(INFO) << "PRINT HOST TENSOR: in";
        //print_tensor_host(in_host);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: out";
        //print_tensor_device(out);
        //sleep(2);

        //LOG(INFO) << "PRINT HOST TENSOR: out_golden";
        //print_tensor_host(out_golden_host);
        //sleep(2);

        double max_r, max_d;
        tensor_cmp_host(out_host.data(), out_golden_host.data(), out_host.size(), max_r, max_d);
        LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << max_r << " max_d = " << max_d;
        LOG(INFO) << "cmp elapse time: device( best : " <<t_device.get_best_ms() <<" ms , average : " << t_device.get_average_ms() << " ms) : host(" << t_host.get_average_ms() << " ms)";
        //CHECK_LE(fabs(statics->max_r), 1.0e-6) << "error result";
        //Verify(out_golden_host, out_host, out_host.size());
        t_device.clear();
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<AMD>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

