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
#include "saber/funcs/base.h"
#include "saber/funcs/impl/amd/saber_softmax.h"
#include "saber/funcs/impl/amd/amd_utils.h"

namespace anakin{
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::init(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD> &param,
        Context<AMD> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}


int nextPow2(int v)
{

    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}


template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::create(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD> &param,
        Context<AMD> &ctx) {

    cl_context context = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    //LOG(INFO) << "device id= " << device << " conext = " << context;

    //To create vld, gld and compile options
    KernelInfo kernerlInfo;

    //To set local work size
    kernerlInfo.l_wk = {256, 1, 1};

    //int num_batch = inputs[0]->num();
    int c = inputs[0]->channel();
    int num_batch = c;

    //To set comp_options
    kernerlInfo.comp_options = std::string(" -DNUM_BATCH=") + std::to_string(num_batch);

    int grid_size = inputs[0]->num() * inputs[0]->width() * inputs[0]->height();

    if(num_batch == 1)
    { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        kernerlInfo.g_wk = {workgroups * kernerlInfo.l_wk[0], 1, 1};
    }
    else
    { // CSR-Stream like approach

        // num_threads iterating over channels for one spatial_dim
        int batch_size = 256 / num_batch;
        // num_channels each threads iterates over to cover all the channels
        int u_batch_size = c > batch_size ? nextPow2(c / batch_size) : 1;

        size_t workgroups =
            grid_size % num_batch == 0 ? grid_size / num_batch : grid_size / num_batch + 1;
        kernerlInfo.g_wk = {workgroups * kernerlInfo.l_wk[0], 1, 1};

        kernerlInfo.comp_options += " -DBATCH_SIZE=" + std::to_string(batch_size) + " -DU_BATCH_SIZE=" +
                 std::to_string(u_batch_size);
    }

    kernerlInfo.kernel_file = "Softmax.cl";
    kernerlInfo.kernel_name = "Softmax";

    //LOG(INFO) << "kernel file name: " << kernerlInfo.kernel_file;
    //LOG(INFO) << "kernel name: " << kernerlInfo.kernel_name;
    //LOG(INFO) << "local work size: " << kernerlInfo.l_wk[0] << " " << kernerlInfo.l_wk[1] << "  " << kernerlInfo.l_wk[2];
    //LOG(INFO) << "global work size: " << kernerlInfo.g_wk[0] << " " << kernerlInfo.g_wk[1] << "  " << kernerlInfo.g_wk[2];
    //LOG(INFO) << "compile option: " << kernerlInfo.comp_options;

    std::copy(kernerlInfo.g_wk.begin(), kernerlInfo.g_wk.end(), _globalWorkSize);
    std::copy(kernerlInfo.l_wk.begin(), kernerlInfo.l_wk.end(), _localWorkSize);

    std::string kernel_file = kernerlInfo.kernel_file;
    std::string kernel_name = kernerlInfo.kernel_name;

    //To create the program
    cl_program program = CreateCLProgram(context, device, kernerlInfo.kernel_file.c_str(), &kernerlInfo);
    if (program == NULL)
    {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel = clCreateKernel(program, kernerlInfo.kernel_name.c_str(), NULL);
    if (_kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }
    
    //LOG(INFO) << "COMPLETE CREATE KERNEL";
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD> &param) {

    //LOG(INFO) << "SaberSoftmax::dispatch";
    cl_int errNum = 0;
    
    //To set the argument
    cl_mem memObjects[1] = { 0 };
    outputs[0]->copy_from(*inputs[0]);
    memObjects[0] = (cl_mem)outputs[0]->data();
    int arg1_channel = inputs[0]->channel();
    int arg2_grid_size = inputs[0]->num() * inputs[0]->width() * inputs[0]->height();
    int arg3_stride_c = inputs[0]->width() * inputs[0]->height();

    errNum = clSetKernelArg(_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(_kernel, 1, sizeof(int), &arg1_channel);
    errNum |= clSetKernelArg(_kernel, 2, sizeof(int), &arg2_grid_size);
    errNum |= clSetKernelArg(_kernel, 3, sizeof(int), &arg3_stride_c);

    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set kernel arguments: " << errNum;
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPLETE SET ARGUMENT";

    //To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    cl_event event;
    errNum = clEnqueueNDRangeKernel(cm, _kernel, 3, NULL,
                                    _globalWorkSize, _localWorkSize,
                                    0, NULL, &event);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set execution: " << errNum;
        return SaberInvalidValue;
    }
    //LOG(INFO) << "COMPLETE EXECUTION";

    cl_event_list list;
    list.push_back(event);
    Env<AMD>::add_event(list);
    return SaberSuccess;
}

template class SaberSoftmax<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, AMD, AK_HALF);

}
} // namespace anakin
