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
#include "saber/funcs/impl/amd/saber_conv_act_pooling.h"
#include "saber/funcs/impl/amd/amd_utils.h"
#include "saber/funcs/conv.h"

namespace anakin{
namespace saber {
#ifdef USE_AMD
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

typedef struct KernelArguType
{
    size_t size;
    void * ptr;
    bool isVal;
} T_KernelArgu;

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DActPooling<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActivePoolingParam<OpTensor> &param,
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
SaberStatus SaberConv2DActPooling<AMD, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ConvActivePoolingParam<OpTensor> &param,
    Context<AMD> &ctx)
{
    LOG(INFO) << "create";

    KernelInfo kernelInfo;
    cl_program program;
    T_ExtSolutionConfig extSolution;

    //NOTE: The width and height of output are parameters for convolution in conv_act_pooling
    std::vector<DataTensor_out *> conv_outputs;
    DataTensor_out* conv_out = new DataTensor_out();
    conv_outputs.push_back(conv_out);
    Conv<AMD, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, conv_outputs, param.conv_param);
    conv_out->re_alloc(conv_out->shape());
    _outConvRelu = conv_out;

    cl_context context = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    //LOG(INFO) << "device id= " << device << " conext = " << context;
    kernelInfo.l_wk = {512, 1, 1};
    kernelInfo.g_wk = {32768, 1, 1};
    kernelInfo.kernel_file = "wino_conv_3x3.so";
    kernelInfo.kernel_name = "sp3AsmConv3x3F";

    //set comp_options...
    kernelInfo.comp_options = "";

    //To create the program
    program = CreatProgramFromBinaryFile(context, device, ("./" + kernelInfo.kernel_file).c_str());

    if (program == NULL)
    {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    std::copy(kernelInfo.g_wk.begin(), kernelInfo.g_wk.end(), _globalWorkSize);
    std::copy(kernelInfo.l_wk.begin(), kernelInfo.l_wk.end(), _localWorkSize);

    //LOG(INFO) << "kernel file name: " << kernelInfo.kernel_file;
    //LOG(INFO) << "kernel name: " << kernelInfo.kernel_name;
    //LOG(INFO) << "local work size: " << kernelInfo.l_wk[0] << " " << kernelInfo.l_wk[1] << "  " << kernelInfo.l_wk[2];
    //LOG(INFO) << "global work size: " << kernelInfo.g_wk[0] << " " << kernelInfo.g_wk[1] << "  " << kernelInfo.g_wk[2];
    //LOG(INFO) << "compile option: " << kernelInfo.comp_options;

    //LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel = clCreateKernel(program, kernelInfo.kernel_name.c_str(), NULL);
    if (_kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPLETE CREATE KERNEL";


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //Start to do pooling...
    switch(param.pooling_param.pooling_type)
    {
        case Pooling_max:
            param.pooling_param.pooling_type = (PoolingType)MLO_POOLING_OP_MAX;
            break;
        default:
            LOG(ERROR) << "Unknown polling type";
            break;
    }

    int _grp_tile0 = 8;
    int _grp_tile1 = 8;
    int _out_width = (_outConvRelu->width() + param.pooling_param.pad_w * 2) / param.pooling_param.window_w;
    int _out_height = (_outConvRelu->height() + param.pooling_param.pad_h * 2) / param.pooling_param.window_h;
    int _out_pix_tile0 = std::max(1, 8 / param.pooling_param.stride_w);
    int _out_pix_tile1 = std::max(1, 8 / param.pooling_param.stride_h);

    while(_out_pix_tile0 * _grp_tile0 > _out_width * 2 && _out_pix_tile0 > 1)
    {
        _out_pix_tile0 >>= 1;
    }

    while(_out_pix_tile1 * _grp_tile1 > _out_height * 2 && _out_pix_tile1 > 1)
    {
        _out_pix_tile1 >>= 1;
    }

    int g_wk_width = ((_out_width + _grp_tile0 * _out_pix_tile0 - 1) /
                      (_grp_tile0 * _out_pix_tile0));
    int g_wk_height = ((_out_height + _grp_tile1 * _out_pix_tile1 - 1) /
                       (_grp_tile1 * _out_pix_tile1));

    //kernelInfo.l_wk = {_grp_tile0, _grp_tile1, 1};
    //kernelInfo.g_wk = {g_wk_width * _grp_tile0, g_wk_height * _grp_tile1, _outConvRelu->channel() * _outConvRelu->num()};
    //kernelInfo.kernel_file = "Pooling.cl";
    //kernelInfo.kernel_name = "Pooling";

    
    kernelInfo.l_wk = {256, 1, 1};
    kernelInfo.g_wk = {64*64*40, 1, 1};
    kernelInfo.kernel_file = "MIOpenBiasReLuPooling.cl";
    kernelInfo.kernel_name = "mloPooling";

    kernelInfo.comp_options =
         std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(param.pooling_param.pooling_type) +
         std::string(" -DMLO_POOLING_KERNEL_SZ0=2") +
         std::string(" -DMLO_POOLING_KERNEL_SZ1=2") +
         std::string(" -DMLO_POOLING_PAD0=") + std::to_string(param.pooling_param.pad_w) +
         std::string(" -DMLO_POOLING_PAD1=") + std::to_string(param.pooling_param.pad_h) +
         std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(param.pooling_param.stride_w) +
         std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(param.pooling_param.stride_h) +
         std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(_outConvRelu->channel()) +
         std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(outputs[0]->channel()) +
         std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(_out_pix_tile0) + //extSolution.horiz_out_pix) +
         std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(_out_pix_tile1) + //extSolution.vert_out_pix) +
         std::string(" -DMLO_POOLING_GROUP_SZ0=8") +
         std::string(" -DMLO_POOLING_GROUP_SZ1=8") +

         std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(_outConvRelu->width()) +
         std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(_outConvRelu->height()) +
         std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(_outConvRelu->width()) +
         std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(_outConvRelu->width() * _outConvRelu->height()) +
         std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(_outConvRelu->width() * _outConvRelu->height() * outputs[0]->channel()) +

         std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string((_outConvRelu->width() + param.pooling_param.pad_w * 2) / param.pooling_param.window_w) +
         std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string((_outConvRelu->height() + param.pooling_param.pad_h * 2) / param.pooling_param.window_h) +
         std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string((_outConvRelu->width() + param.pooling_param.pad_w * 2) / param.pooling_param.window_w) +
         std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string((_outConvRelu->width() * _outConvRelu->height()) / (param.pooling_param.window_w * param.pooling_param.window_h)) +
         std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string((_outConvRelu->width() * _outConvRelu->height() * outputs[0]->channel()) / (param.pooling_param.window_w * param.pooling_param.window_h)) +
         std::string(" -DBATCH_NUM=") + std::to_string(inputs[0]->num()) +
         std::string(" -DCU_NUM=64");


    //LOG(INFO) << "kernel file name: " << kernelInfo.kernel_file;
    //LOG(INFO) << "kernel name: " << kernelInfo.kernel_name;
    //LOG(INFO) << "local work size: " << kernelInfo.l_wk[0] << " " << kernelInfo.l_wk[1] << "  " << kernelInfo.l_wk[2];
    //LOG(INFO) << "global work size: " << kernelInfo.g_wk[0] << " " << kernelInfo.g_wk[1] << "  " << kernelInfo.g_wk[2];
    //LOG(INFO) << "compile option: " << kernelInfo.comp_options;
  
    //////////////////////////////////////////////////
    std::copy(kernelInfo.g_wk.begin(), kernelInfo.g_wk.end(), _globalWorkSize2);
    std::copy(kernelInfo.l_wk.begin(), kernelInfo.l_wk.end(), _localWorkSize2);

    std::string kernel_file = kernelInfo.kernel_file;
    std::string kernel_name = kernelInfo.kernel_name;

    //To create the program
    cl_program program2 = CreateCLProgram(context, device, kernelInfo.kernel_file.c_str(), &kernelInfo);
    if (program == NULL)
    {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel2 = clCreateKernel(program2, kernelInfo.kernel_name.c_str(), NULL);
    if (_kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPLETE CREATE KERNEL";
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DActPooling<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActivePoolingParam<OpTensor> &param)
{
    //LOG(INFO) << "dispatch";

    cl_int errNum = 0;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    //To set the argument
    cl_uint uintObjects[8] = {0, 0, 0, 0,
                            0, 0, 0, 0};
    cl_mem memObjects[4] = {0, 0, 0, 0};

    const ClMem* clin;
    ClMem* clout;
    const ClMem* clweight;
    const ClMem* clbias;

    size_t offset_in, offset_out, offset_weight, offset_bias;

    int d_n_groups = 64, d_flags = 0;

    uintObjects[0] = (cl_uint)inputs[0]->num();
    uintObjects[1] = (cl_uint)inputs[0]->channel();
    uintObjects[2] = (cl_uint)inputs[0]->height();
    uintObjects[3] = (cl_uint)inputs[0]->width();
    uintObjects[4] = (cl_uint)param.conv_param.weight()->num();
    uintObjects[5] = d_n_groups;
    uintObjects[6] = d_flags;
    uintObjects[7] = 0;


    clin = inputs[0]->data();
    clweight = param.conv_param.weight()->data();
    clout = _outConvRelu->mutable_data();
    offset_in = clin->offset;
    offset_weight = clweight->offset;
    offset_out = clout->offset;

    memObjects[0] = clin->dmem;//(cl_mem)inputs[0]->data();
    memObjects[1] = clweight->dmem;//(cl_mem)param.conv_param.weight()->data();
    memObjects[2] = clout->dmem;//(cl_mem)_outConvRelu->mutable_data();
    //memObjects[3] = (cl_mem)d_return_addr;

    errNum = setKernelArgs(_kernel, uintObjects[0], uintObjects[1], uintObjects[2],
                            uintObjects[3], uintObjects[4], uintObjects[5], 
                            uintObjects[6], uintObjects[7], memObjects[0],
                            memObjects[1], memObjects[2], memObjects[3]);

    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set kernel arguments";
        return SaberInvalidValue;
    }
    //LOG(INFO) << "COMPLETE SET ARGUMENT";

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

    ////////////////////////////////////////////////////////////////////////////////////////

    //To set the argument
    cl_mem memObjects2[3] = { 0, 0, 0};

    clin = _outConvRelu->data();
    clout = outputs[0]->mutable_data();
    clbias = param.conv_param.bias()->data();
    offset_in = clin->offset;
    offset_out = clout->offset;
    offset_bias = clbias->offset;

    memObjects2[0] = clin->dmem;//(cl_mem)_outConvRelu->data();
    memObjects2[1] = clout->dmem;//(cl_mem)outputs[0]->mutable_data();
    memObjects2[2] = clbias->dmem;//(cl_mem)param.conv_param.bias()->data();

    //errNum = clSetKernelArg(_kernel2, 0, sizeof(cl_mem), &memObjects2[0]);
    //errNum |= clSetKernelArg(_kernel2, 1, sizeof(cl_mem), &memObjects2[1]);
    errNum = setKernelArgs(_kernel2, memObjects2[0], memObjects2[1], memObjects2[2], param.activation_param.negative_slope);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set kernel arguments";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPLETE SET ARGUMENT";
    cl_event event2;
    errNum = clEnqueueNDRangeKernel(cm, _kernel2, 3, NULL,
                                    _globalWorkSize2, _localWorkSize2,
                                    0, NULL, &event2);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set execution: " << errNum;
        return SaberInvalidValue;
    }
    //LOG(INFO) << "COMPLETE EXECUTION";
    cl_event_list list;
    list.push_back(event);
    list.push_back(event2);
    Env<AMD>::add_event(list);
    return SaberSuccess;
}
#endif
}
} // namespace anakin
