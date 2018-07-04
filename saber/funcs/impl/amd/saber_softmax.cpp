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

#include "saber/funcs/impl/amd/saber_softmax.h"
#include "saber/funcs/impl/amd/amd_utils.h"

namespace anakin{
namespace saber{
#ifdef USE_AMD
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD, AK_FLOAT, NCHW> TensorDf4;

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberSoftmax<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<OpTensor> &param,
        Context<AMD> &ctx)
{

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberSoftmax<AMD, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SoftmaxParam<OpTensor> &param,
    Context<AMD> &ctx)
{

    cl_context context = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    LOG(INFO) << "device id= " << device << " conext = " << context;

    //To create vld, gld and compile options
    KernelInfo construction_params;

    int grid_size = inputs[0]->num() * inputs[0]->width() * inputs[0]->height();
    size_t workgroups = std::min(grid_size, 64 * 40 * 8);

    construction_params.l_wk = {256, 1, 1};
    construction_params.g_wk = {workgroups * construction_params.l_wk[0], 1, 1};
    construction_params.kernel_file = "Softmax.cl";
    construction_params.kernel_name = "Softmax";

    //To set comp_options
    construction_params.comp_options = std::string(" -DNUM_BATCH=1");

    LOG(INFO) << "kernel file name: " << construction_params.kernel_file;
    LOG(INFO) << "kernel name: " << construction_params.kernel_name;
    LOG(INFO) << "local work size: " << construction_params.l_wk[0] << " " << construction_params.l_wk[1] << "  " << construction_params.l_wk[2];
    LOG(INFO) << "global work size: " << construction_params.g_wk[0] << " " << construction_params.g_wk[1] << "  " << construction_params.g_wk[2];
    LOG(INFO) << "compile option: " << construction_params.comp_options;

    std::copy(construction_params.g_wk.begin(), construction_params.g_wk.end(), _globalWorkSize);
    std::copy(construction_params.l_wk.begin(), construction_params.l_wk.end(), _localWorkSize);

    std::string kernel_file = construction_params.kernel_file;
    std::string kernel_name = construction_params.kernel_name;

    //To create the program
    cl_program program = CreateCLProgram(context, device, construction_params.kernel_file.c_str(), &construction_params);
    if (program == NULL)
    {
        LOG(INFO) << "Failed to load program";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel = clCreateKernel(program, construction_params.kernel_name.c_str(), NULL);
    if (_kernel == NULL)
    {
        LOG(INFO) << "Failed to create kernel";
        return SaberInvalidValue;
    }
    
    LOG(INFO) << "COMPLETE CREATE KERNEL";
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberSoftmax<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<OpTensor> &param)
{

    LOG(INFO) << "SaberSoftmax::dispatch";

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    Shape stride_in = inputs[0]->get_stride();
    Shape stride_out = outputs[0]->get_stride();

    cl_int errNum = 0;
    
    //To set the argument
    cl_mem memObjects[1] = { 0 };

    const ClMem clin;
    ClMem clout;

    size_t offset_in, offset_out;

    clin = inputs[0]->data();
    clout = outputs[0]->mutable_data();

    offset_in = clin.offset;
    offset_out = clout.offset;

    outputs[0]->copy_from(*inputs[0]);
    memObjects[0] = clout.dmem;//(cl_mem)outputs[0]->data();
    int arg1_channel = inputs[0]->channel();
    int arg2_grid_size = inputs[0]->num() * inputs[0]->width() * inputs[0]->height();
    int arg3_stride_c = inputs[0]->width() * inputs[0]->height();

    errNum = clSetKernelArg(_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(_kernel, 1, sizeof(int), &arg1_channel);
    errNum |= clSetKernelArg(_kernel, 2, sizeof(int), &arg2_grid_size);
    errNum |= clSetKernelArg(_kernel, 3, sizeof(int), &arg3_stride_c);

    if (errNum != CL_SUCCESS)
    {
        LOG(INFO) << "Fail to set kernel arguments";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPLETE SET ARGUMENT";

    //To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    errNum = clEnqueueNDRangeKernel(cm, _kernel, 3, NULL,
                                    _globalWorkSize, _localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        LOG(INFO) << "Fail to set execution: " << errNum;
        return SaberInvalidValue;
    }
    LOG(INFO) << "COMPLETE EXECUTION";

    return SaberSuccess;
}
#endif
}
}

