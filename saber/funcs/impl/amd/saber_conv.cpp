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
#include "saber/funcs/impl/amd/saber_conv.h"
#include "saber/funcs/impl/amd/amd_utils.h"
#include <list>

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
SaberStatus SaberConv2D<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvParam<OpTensor> &param,
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
SaberStatus SaberConv2D<AMD, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ConvParam<OpTensor> &param,
    Context<AMD> &ctx)
{
    LOG(INFO) << "create";

    KernelInfo kernelInfo;
    cl_program program;
    T_ExtSolutionConfig extSolution;

    cl_context context = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    //LOG(INFO) << "device id= " << device << " conext = " << context;

    //Set Solver and get solution back
    if (inputs[0]->num() == 1 && inputs[0]->channel() == 3 && 
        inputs[0]->height() == 224 && inputs[0]->width() == 224) {
        kernelInfo.l_wk = {256, 1, 1};
        kernelInfo.g_wk = {12544, 8, 1};
        kernelInfo.kernel_file = "ConvFwd3x3.cl";
        kernelInfo.kernel_name = "ConvFwd3x3";

        extSolution.in_tile0 = 32;
        extSolution.in_tile1 = 32;
        extSolution.grp_tile0 = 16;
        extSolution.grp_tile1 = 16;
        extSolution.out_pix_tile0 = 2;
        extSolution.out_pix_tile1 = 2;
        extSolution.n_stacks = 1;
        extSolution.n_out_pix_tiles = 8;
        extSolution.n_out_tiles_perstack = 8;
        extSolution.n_in_data_tiles = 2;
        extSolution.n_read_procs = 256;
        extSolution.alu_tile0 = 16;
        extSolution.alu_tile1 = 16;

        //set comp_options...
        kernelInfo.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=64") +	// (fixed) wave=64
        std::string(" -DMLO_DIR_FORWARD=1") +	// (fixed) forward
        std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(param.weight()->width()) +
        std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(param.weight()->height()) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(param.pad_w) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(param.pad_h) +
        std::string(" -DMLO_FILTER_STRIDE0=1") + // (fixed temp)
        std::string(" -DMLO_FILTER_STRIDE1=1") +// (fixed temp)
        std::string(" -DMLO_N_OUTPUTS=") + std::to_string(outputs[0]->channel()) +
        std::string(" -DMLO_N_INPUTS=") + std::to_string(inputs[0]->channel()) +
        std::string(" -DMLO_BATCH_SZ=") + std::to_string(inputs[0]->num()) +
        std::string(" -DMLO_OUT_WIDTH=") + std::to_string(outputs[0]->width()) +
        std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(outputs[0]->height()) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(outputs[0]->width() * outputs[0]->height() * outputs[0]->channel()) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(outputs[0]->width() * outputs[0]->height()) +
        std::string(" -DMLO_OUT_STRIDE=") + std::to_string(outputs[0]->width()) +
        std::string(" -DMLO_IN_WIDTH=") + std::to_string(inputs[0]->width()) +
        std::string(" -DMLO_IN_HEIGHT=") + std::to_string(inputs[0]->height()) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(inputs[0]->width() * inputs[0]->height() * inputs[0]->channel()) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(inputs[0]->width() * inputs[0]->height()) +
        std::string(" -DMLO_IN_STRIDE=") + std::to_string(inputs[0]->width()) +
        std::string(" -DMLO_IN_TILE0=") + std::to_string(extSolution.in_tile0) +
        std::string(" -DMLO_IN_TILE1=") + std::to_string(extSolution.in_tile1) +
        std::string(" -DMLO_GRP_TILE0=") + std::to_string(extSolution.grp_tile0) +
        std::string(" -DMLO_GRP_TILE1=") + std::to_string(extSolution.grp_tile1) +
        std::string(" -DMLO_OUT_TILE0=") + std::to_string(extSolution.out_pix_tile0) +
        std::string(" -DMLO_OUT_TILE1=") + std::to_string(extSolution.out_pix_tile1) +
        std::string(" -DMLO_N_STACKS=") + std::to_string(extSolution.n_stacks) +
        std::string(" -DMLO_N_OUT_TILES=") + std::to_string(extSolution.n_out_pix_tiles) +
        std::string(" -DMLO_N_OUT_TILES_PERSTACK=") + std::to_string(extSolution.n_out_tiles_perstack) +
        std::string(" -DMLO_N_IN_TILES_PERSTACK=") + std::to_string(extSolution.n_in_data_tiles) +
        std::string(" -DMLO_N_READ_PROCS=") + std::to_string(extSolution.n_read_procs) +
        std::string(" -DMLO_CONV_BIAS=0") + // (for now not support)
        std::string(" -DMLO_ALU_VTILE0=") + std::to_string(extSolution.alu_tile0) +
        std::string(" -DMLO_ALU_VTILE1=") + std::to_string(extSolution.alu_tile1);

        program = CreateCLProgram(context, device, kernelInfo.kernel_file.c_str(), &kernelInfo);
    } else {
        kernelInfo.l_wk = {512, 1, 1};
        kernelInfo.g_wk = {32768, 1, 1};
        kernelInfo.kernel_file = "wino_conv_3x3.so";
        kernelInfo.kernel_name = "sp3AsmConv3x3F";

        //set comp_options...
        kernelInfo.comp_options = "";

        //To create the program
        program = CreatProgramFromBinaryFile(context, device, ("./" + kernelInfo.kernel_file).c_str());
    }
    if (program == NULL)
    {
        LOG(INFO) << "Failed to load program";
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
        LOG(INFO) << "Failed to create kernel";
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
SaberStatus SaberConv2D<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvParam<OpTensor> &param)
{
    //LOG(INFO) << "dispatch";

    cl_int errNum = 0;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    //To set the argument
    if (inputs[0]->num() == 1 && inputs[0]->channel() == 3 &&
        inputs[0]->height() == 224 && inputs[0]->width() == 224) {
        cl_mem memObjects[3] = { 0, 0, 0 };

        const ClMem clin = inputs[0]->data();
        ClMem clout = outputs[0]->mutable_data();
        const ClMem clweight = param.weight()->data();

        size_t offset_in = clin.offset;
        size_t offset_out = clout.offset;
        size_t offset_weight = clweight.offset;

        memObjects[0] = clin.dmem;//(cl_mem)inputs[0]->data();
        memObjects[1] = clweight.dmem;//(cl_mem)param.weight()->data();
        memObjects[2] = clout.dmem;//(cl_mem)outputs[0]->mutable_data();

        errNum = setKernelArgs(_kernel, memObjects[0], memObjects[1], memObjects[2]);
        if (errNum != CL_SUCCESS)
        {
            LOG(INFO) << "Fail to set kernel arguments";
            return SaberInvalidValue;
        }
        //LOG(INFO) << "COMPLETE SET ARGUMENT";
    } else {
        cl_uint uintObjects[8] = {0, 0, 0, 0,
                                0, 0, 0, 0};
        cl_mem memObjects[4] = {0, 0, 0, 0};
        int d_n_groups = 64, d_flags = 0, d_reserved = 0;

        uintObjects[0] = (cl_uint)inputs[0]->num();
        uintObjects[1] = (cl_uint)inputs[0]->channel();
        uintObjects[2] = (cl_uint)inputs[0]->height();
        uintObjects[3] = (cl_uint)inputs[0]->width();
        uintObjects[4] = (cl_uint)param.weight()->num();
        uintObjects[5] = d_n_groups;
        uintObjects[6] = d_flags;
        uintObjects[7] = d_reserved;

        const ClMem clin = inputs[0]->data();
        ClMem clout = outputs[0]->mutable_data();
        const ClMem clweight = param.weight()->data();

        size_t offset_in = clin.offset;
        size_t offset_out = clout.offset;
        size_t offset_weight = clweight.offset;

        memObjects[0] = clin.dmem;//(cl_mem)inputs[0]->data();
        memObjects[1] = clweight.dmem;//(cl_mem)param.weight()->data();
        memObjects[2] = clout.dmem;//(cl_mem)outputs[0]->mutable_data();

        //memObjects[0] = (cl_mem)inputs[0]->data();
        //memObjects[1] = (cl_mem)param.weight()->data();
        //memObjects[2] = (cl_mem)outputs[0]->mutable_data();
        //memObjects[3] = (cl_mem)d_return_addr;

		errNum = setKernelArgs(_kernel, uintObjects[0], uintObjects[1], uintObjects[2],
                                uintObjects[3], uintObjects[4], uintObjects[5], 
                                uintObjects[6], uintObjects[7], memObjects[0],
                                memObjects[1], memObjects[2], memObjects[3]);

        if (errNum != CL_SUCCESS)
        {
            LOG(INFO) << "Fail to set kernel arguments";
            return SaberInvalidValue;
        }

        //LOG(INFO) << "COMPLETE SET ARGUMENT";
    }

    errNum = clEnqueueNDRangeKernel(cm, _kernel, 3, NULL,
                                    _globalWorkSize, _localWorkSize,
                                    0, NULL, NULL); 
    if (errNum != CL_SUCCESS)
    {
        LOG(INFO) << "Fail to set execution: " << errNum;
        return SaberInvalidValue;
    }
    //LOG(INFO) << "COMPLETE EXECUTION";

    return SaberSuccess;
}
#endif
}
} // namespace anakin
