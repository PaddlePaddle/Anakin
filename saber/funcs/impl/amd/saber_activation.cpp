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
#include "saber/funcs/impl/amd/saber_activation.h"
#include "saber/funcs/impl/amd/amd_utils.h"

namespace anakin{
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::init(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ActivationParam<AMD> &param,
        Context<AMD> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::create(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ActivationParam<AMD> &param,
        Context<AMD> &ctx) {

    this->_ctx = &ctx;

    cl_context context = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    //LOG(INFO) << "device id= " << device << " conext = " << context;

    KernelInfo kernelInfo;

    switch (param.active){
        case Active_relu:
            //TODO
            //Rewrite here once solver is ready.//////////////
            T_ExtSolutionConfig extSolution;
            //LOG(INFO) << inputs[0]->width() << " x " << inputs[0]->height() << " x " << inputs[0]->channel();
            switch(inputs[0]->width())
            {
                case 224:
                    kernelInfo.l_wk = {256, 1, 1};
                    kernelInfo.g_wk = {12544, 8, 1};

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
                    break;

                case 112:
                    kernelInfo.l_wk = {256, 1, 1};
                    kernelInfo.g_wk = {4096, 16, 1};

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
                    break;

                case 56:
                    kernelInfo.l_wk = {256, 1, 1};
                    kernelInfo.g_wk = {1024, 32, 1};

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
                    break;

                case 28:
                    kernelInfo.l_wk = {256, 1, 1};
                    kernelInfo.g_wk = {256, 64, 1};

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
                    break;
                case 14:
                    kernelInfo.l_wk = {64, 1, 1};
                    kernelInfo.g_wk = {64, 64, 1};

                    extSolution.in_tile0 = 16;
                    extSolution.in_tile1 = 16;
                    extSolution.grp_tile0 = 8;
                    extSolution.grp_tile1 = 8;
                    extSolution.out_pix_tile0 = 2;
                    extSolution.out_pix_tile1 = 2;
                    extSolution.n_stacks = 1;
                    extSolution.n_out_pix_tiles = 8;
                    extSolution.n_out_tiles_perstack = 8;
                    extSolution.n_in_data_tiles = 2;
                    extSolution.n_read_procs = 64;
                    extSolution.alu_tile0 = 8;
                    extSolution.alu_tile1 = 8;
                    break;
                case 1:
                    if (inputs[0]->channel() == 4096) {
                        kernelInfo.l_wk = {256, 1, 1};
                        kernelInfo.g_wk = {4096, 1, 1};
                    }else if(inputs[0]->channel()  == 1000) {
                        kernelInfo.l_wk = {256, 1, 1};
                        kernelInfo.g_wk = {1024, 1, 1};
                    }
                    break;
            }
            
            kernelInfo.comp_options =
                std::string(" -DMLO_HW_WAVE_SZ=64") +    // (fixed) wave=64
                std::string(" -DMLO_DIR_FORWARD=1") +    // (fixed) forward
                std::string(" -DMLO_FILTER_STRIDE0=1") + // (fixed temp)
                std::string(" -DMLO_FILTER_STRIDE1=1") + // (fixed temp)
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
                std::string(" -DMLO_CONV_BIAS=0") ; // (for now not support)

            if(inputs[0]->width() == 1) {

                kernelInfo.kernel_file = "ReluUni.cl";
                kernelInfo.kernel_name = "ReluUni";

            } else  {
            //set comp_options...

                kernelInfo.comp_options +=
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
                    std::string(" -DMLO_ALU_VTILE0=") + std::to_string(extSolution.alu_tile0) +
                    std::string(" -DMLO_ALU_VTILE1=") + std::to_string(extSolution.alu_tile1);

                kernelInfo.kernel_file = "Relu.cl";
                kernelInfo.kernel_name = "Relu";
             }
            break;

        case Active_sigmoid:

            break;

        case Active_tanh:

            break;

        case Active_clipped_relu:

            break;

        case Active_elu:

            break;
    }
    std::copy(kernelInfo.g_wk.begin(), kernelInfo.g_wk.end(), _globalWorkSize);
    std::copy(kernelInfo.l_wk.begin(), kernelInfo.l_wk.end(), _localWorkSize);

    //LOG(INFO) << "kernel file name: " << kernelInfo.kernel_file;
    //LOG(INFO) << "kernel name: " << kernelInfo.kernel_name;
    //LOG(INFO) << "local work size: " << kernelInfo.l_wk[0] << " " << kernelInfo.l_wk[1] << "  " << kernelInfo.l_wk[2];
    //LOG(INFO) << "global work size: " << kernelInfo.g_wk[0] << " " << kernelInfo.g_wk[1] << "  " << kernelInfo.g_wk[2];
    //LOG(INFO) << "compile option: " << kernelInfo.comp_options;

    //To create the program
    cl_program program = CreateCLProgram(context, device, kernelInfo.kernel_file.c_str(), &kernelInfo);
    if (program == NULL)
    {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPILE OCL KERNEL CODE";

    //To create kernel
    _kernel = clCreateKernel(program, kernelInfo.kernel_name.c_str(), NULL);
    if (_kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    //LOG(INFO) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ActivationParam<AMD> &param) {

    cl_int errNum = 0;
    //To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    //To set the argument
    cl_mem memObjects[2] = { 0, 0 };
    memObjects[0] = (cl_mem)inputs[0]->data();
    memObjects[1] = (cl_mem)outputs[0]->mutable_data();

    errNum = clSetKernelArg(_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(_kernel, 2, sizeof(float), &param.negative_slope);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Fail to set kernel arguments";
        return SaberInvalidValue;
    }
    cl_event event;
    //LOG(INFO) << "COMPLETE SET ARGUMENT";
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

template class SaberActivation<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, AMD, AK_HALF);

}
} // namespace anakin
