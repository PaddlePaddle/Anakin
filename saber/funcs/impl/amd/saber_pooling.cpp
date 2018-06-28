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


#include "saber/funcs/impl/amd/saber_pooling.h"
#include "saber/funcs/impl/amd/amd_utils.h"
#include "saber/funcs/conv.h"


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
SaberStatus SaberPooling<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        PoolingParam<OpTensor> &param,
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
SaberStatus SaberPooling<AMD, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    PoolingParam<OpTensor> &param,
    Context<AMD> &ctx)
{

    cl_context context = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    LOG(INFO) << "device id= " << device << " conext = " << context;

    //Set Solver and get solution back
    //TODO

    switch(param.pooling_type)
    {
        case Pooling_max:
            param.pooling_type = (PoolingType)MLO_POOLING_OP_MAX;
            break;
        default:
            LOG(INFO) << "Unknown polling type";
            break;
    }

    KernelInfo kernelInfo;

    //TODO
    //Rewrite here once solver is ready.//////////////
    T_ExtSolutionConfig extSolution;
    //////////////////////////////////////////////////
    int _grp_tile0 = 8;
    int _grp_tile1 = 8;
    int _out_width = (inputs[0]->width() + param.pad_w * 2) / param.window_w;
    int _out_height = (inputs[0]->height() + param.pad_h * 2) / param.window_h;
    int _out_pix_tile0 = std::max(1, 8 / param.stride_w);
    int _out_pix_tile1 = std::max(1, 8 / param.stride_h);

    while(_out_pix_tile0 * _grp_tile0 > _out_width * 2 && _out_pix_tile0 > 1)
    {
        _out_pix_tile0 >>= 1;
    }

    while(_out_pix_tile1 * _grp_tile1 > _out_height * 2 && _out_pix_tile1 > 1)
    {
        _out_pix_tile1 >>= 1;
    }

    //int g_wk_width = ((_out_width + _grp_tile0 * _out_pix_tile0 - 1) /
    //                  (_grp_tile0 * _out_pix_tile0));
    //int g_wk_height = ((_out_height + _grp_tile1 * _out_pix_tile1 - 1) /
    //                   (_grp_tile1 * _out_pix_tile1));

    //kernelInfo.l_wk = {_grp_tile0, _grp_tile1, 1};
    //kernelInfo.g_wk = {g_wk_width * _grp_tile0, g_wk_height * _grp_tile1, inputs[0]->channel() * inputs[0]->num()};
    //kernelInfo.kernel_file = "Pooling.cl";
    //kernelInfo.kernel_name = "Pooling";
   
    kernelInfo.l_wk = {256, 1, 1};
    kernelInfo.g_wk = {64*64*40, 1, 1};
    kernelInfo.kernel_file = "Pooling.cl";
    kernelInfo.kernel_name = "mloPooling";

    //set comp_options...
    kernelInfo.comp_options =
         std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(param.pooling_type) +
         std::string(" -DMLO_POOLING_KERNEL_SZ0=2") +
         std::string(" -DMLO_POOLING_KERNEL_SZ1=2") +
         std::string(" -DMLO_POOLING_PAD0=") + std::to_string(param.pad_w) +
         std::string(" -DMLO_POOLING_PAD1=") + std::to_string(param.pad_h) +
         std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(param.stride_w) +
         std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(param.stride_h) +
         std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(inputs[0]->channel()) +
         std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(outputs[0]->channel()) +
         std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(_out_pix_tile0) + //extSolution.horiz_out_pix) +
         std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(_out_pix_tile1) + //extSolution.vert_out_pix) +
         std::string(" -DMLO_POOLING_GROUP_SZ0=8") +
         std::string(" -DMLO_POOLING_GROUP_SZ1=8") +

         std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(inputs[0]->width()) +
         std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(inputs[0]->height()) +
         std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(inputs[0]->width()) +
         std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(inputs[0]->width() * inputs[0]->height()) +
         std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(inputs[0]->width() * inputs[0]->height() * outputs[0]->channel()) +

         std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string((inputs[0]->width() + param.pad_w * 2) / param.window_w) +
         std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string((inputs[0]->height() + param.pad_h * 2) / param.window_h) +
         std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string((inputs[0]->width() + param.pad_w * 2) / param.window_w) +
         std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string((inputs[0]->width() * inputs[0]->height()) / (param.window_w * param.window_h)) +
         std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string((inputs[0]->width() * inputs[0]->height() * outputs[0]->channel()) / (param.window_w * param.window_h)) +
         std::string(" -DBATCH_NUM=") + std::to_string(inputs[0]->num()) +
         std::string(" -DCU_NUM=64");

    LOG(INFO) << "kernel file name: " << kernelInfo.kernel_file;
    LOG(INFO) << "kernel name: " << kernelInfo.kernel_name;
    LOG(INFO) << "local work size: " << kernelInfo.l_wk[0] << " " << kernelInfo.l_wk[1] << "  " << kernelInfo.l_wk[2];
    LOG(INFO) << "global work size: " << kernelInfo.g_wk[0] << " " << kernelInfo.g_wk[1] << "  " << kernelInfo.g_wk[2];
    LOG(INFO) << "compile option: " << kernelInfo.comp_options;
  
    //////////////////////////////////////////////////
    std::copy(kernelInfo.g_wk.begin(), kernelInfo.g_wk.end(), _globalWorkSize);
    std::copy(kernelInfo.l_wk.begin(), kernelInfo.l_wk.end(), _localWorkSize);

    std::string kernel_file = kernelInfo.kernel_file;
    std::string kernel_name = kernelInfo.kernel_name;

    //To create the program
    cl_program program = CreateCLProgram(context, device, kernelInfo.kernel_file.c_str(), &kernelInfo);
    if (program == NULL)
    {
        LOG(INFO) << "Failed to load program";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel = clCreateKernel(program, kernelInfo.kernel_name.c_str(), NULL);
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
SaberStatus SaberPooling<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        PoolingParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
 
    LOG(INFO) << "SaberAMDPooling::dispatch";

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    Shape stride_in = inputs[0]->get_stride();
    Shape stride_out = outputs[0]->get_stride();

    cl_int errNum = 0;

    //To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    //To set the argument
    cl_mem memObjects[2] = { 0, 0 };
    memObjects[0] = (cl_mem)inputs[0]->data();
    memObjects[1] = (cl_mem)outputs[0]->mutable_data();

    errNum = clSetKernelArg(_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    if (errNum != CL_SUCCESS)
    {
        LOG(INFO) << "Fail to set kernel arguments";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPLETE SET ARGUMENT";

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
