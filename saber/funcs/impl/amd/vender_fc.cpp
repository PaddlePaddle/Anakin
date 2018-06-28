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

#include "saber/funcs/impl/amd/vender_fc.h"
#include "saber/funcs/impl/amd/amd_utils.h"
#include "saber/funcs/timer.h" 
namespace anakin{
namespace saber {
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
SaberStatus VenderFc<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<AMD> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS1   (128)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS1   (8192*8)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS2   (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS2   (8192*4)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS4   (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS4   (8192*8)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS8   (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS8   (8192*4)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS32  (256)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS32  (8192*4)

#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS1   (256)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS1   (8192*2)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS2   (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS2   (8192*4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS4   (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS4   (8192*4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS8   (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS8   (8192*4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS32  (256)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS32  (8192*4)

#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS1   (256)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS1   (8192*2)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS2   (128)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS2   (8192*4)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS4   (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS4   (8192*4)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS8   (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS8   (8192*8)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS32  (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS32  (8192*8)

#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC6M1.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC6M2.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC6M4.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC6M8.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC6M32.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC7M1.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC7M2.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC7M4.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC7M8.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC7M32.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC8M1.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC8M2.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC8M4.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC8M8.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC8M32.cl"

#define BATCH_SIZE_1_INDEX 0
#define BATCH_SIZE_2_INDEX 1
#define BATCH_SIZE_4_INDEX 2
#define BATCH_SIZE_8_INDEX 3
#define BATCH_SIZE_32_INDEX 4

#define FC6_INDEX 0
#define FC7_INDEX 1
#define FC8_INDEX 2

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::create(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<AMD> &ctx)
{
    this->_ctx = &ctx;
    this->_param = &param;

    cl_device_id device = 0;
    cl_context context = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; //anakin device id to AMD device
    device = dev.get_device();
    context = dev.get_context();

    KernelInfo kernelInfo;
    int batch_size_index = 0;
    int fc_index = 0;

    int M = inputs[0]->num();
    int K = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    int N = param.num_output;
    if (N <= 0) {
        int weight_size = param.weights->valid_size();
        N = weight_size / K;
    }

    int gwk[5][3] = {
            {VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS1, VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS1, VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS1},
            {VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS2, VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS2, VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS2},
            {VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS4, VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS4, VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS4},
            {VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS8, VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS8, VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS8},
            {VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS32, VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS32, VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS32}};
    int lwk[5][3] = {
            {VGG16_FC6_NT_LOCAL_WORK_SIZE_BS1, VGG16_FC7_NT_LOCAL_WORK_SIZE_BS1, VGG16_FC8_NT_LOCAL_WORK_SIZE_BS1},
            {VGG16_FC6_NT_LOCAL_WORK_SIZE_BS2, VGG16_FC7_NT_LOCAL_WORK_SIZE_BS2, VGG16_FC8_NT_LOCAL_WORK_SIZE_BS2},
            {VGG16_FC6_NT_LOCAL_WORK_SIZE_BS4, VGG16_FC7_NT_LOCAL_WORK_SIZE_BS4, VGG16_FC8_NT_LOCAL_WORK_SIZE_BS4},
            {VGG16_FC6_NT_LOCAL_WORK_SIZE_BS8, VGG16_FC7_NT_LOCAL_WORK_SIZE_BS8, VGG16_FC8_NT_LOCAL_WORK_SIZE_BS8},
            {VGG16_FC6_NT_LOCAL_WORK_SIZE_BS32, VGG16_FC7_NT_LOCAL_WORK_SIZE_BS32, VGG16_FC8_NT_LOCAL_WORK_SIZE_BS32}};
    const std::string kfn[5][3] = {
            {VGG16_FC6_NT_KERNEL_FILE_NAME_BS1, VGG16_FC7_NT_KERNEL_FILE_NAME_BS1, VGG16_FC8_NT_KERNEL_FILE_NAME_BS1},
            {VGG16_FC6_NT_KERNEL_FILE_NAME_BS2, VGG16_FC7_NT_KERNEL_FILE_NAME_BS2, VGG16_FC8_NT_KERNEL_FILE_NAME_BS2},
            {VGG16_FC6_NT_KERNEL_FILE_NAME_BS4, VGG16_FC7_NT_KERNEL_FILE_NAME_BS4, VGG16_FC8_NT_KERNEL_FILE_NAME_BS4},
            {VGG16_FC6_NT_KERNEL_FILE_NAME_BS8, VGG16_FC7_NT_KERNEL_FILE_NAME_BS8, VGG16_FC8_NT_KERNEL_FILE_NAME_BS8},
            {VGG16_FC6_NT_KERNEL_FILE_NAME_BS32, VGG16_FC7_NT_KERNEL_FILE_NAME_BS32, VGG16_FC8_NT_KERNEL_FILE_NAME_BS32}};

    switch(inputs[0]->num()) {
        case 1: batch_size_index = BATCH_SIZE_1_INDEX; break;
        case 2: batch_size_index = BATCH_SIZE_2_INDEX; break;
        case 4: batch_size_index = BATCH_SIZE_4_INDEX; break;
        case 8: batch_size_index = BATCH_SIZE_8_INDEX; break;
        case 32: batch_size_index = BATCH_SIZE_32_INDEX; break;
    }

    switch(param.weights->width()*param.weights->height()) {
        case 25088*4096: fc_index = FC6_INDEX; break;
        case 4096*4096: fc_index = FC7_INDEX; break;
        case 4096*1000: fc_index = FC8_INDEX; break;
    }

    if (!param.is_transpose_weights) {
        kernelInfo.l_wk = {lwk[batch_size_index][fc_index], 1, 1};
        kernelInfo.g_wk = {gwk[batch_size_index][fc_index], 1, 1};
        kernelInfo.kernel_file = kfn[batch_size_index][fc_index];
        kernelInfo.kernel_name = "InnerProduct";
    } else {
        LOG(ERROR) << "NOT IMPLEMENT!";
    }

    //set comp_options...
    kernelInfo.comp_options =
        std::string(" -DM=") + std::to_string(M) +
        std::string(" -DN=") + std::to_string(N) +
        std::string(" -DK=") + std::to_string(K);

    //LOG(INFO) << "kernel file name: " << kernelInfo.kernel_file;
    //LOG(INFO) << "kernel name: " << kernelInfo.kernel_name;
    //LOG(INFO) << "local work size: " << kernelInfo.l_wk[0] << " " << kernelInfo.l_wk[1] << "  " << kernelInfo.l_wk[2];
    //LOG(INFO) << "global work size: " << kernelInfo.g_wk[0] << " " << kernelInfo.g_wk[1] << "  " << kernelInfo.g_wk[2];
    //LOG(INFO) << "compile option: " << kernelInfo.comp_options;

    std::copy(kernelInfo.g_wk.begin(), kernelInfo.g_wk.end(), _globalWorkSize);
    std::copy(kernelInfo.l_wk.begin(), kernelInfo.l_wk.end(), _localWorkSize);

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

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<AMD, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param)
{
//    LOG(INFO) << "dispatch";
//    LOG(INFO) << "num_output=" << param.num_output << "axis=" << param.axis;
//    LOG(INFO) << "num=" << inputs[0]->num() << " channel=" << inputs[0]->channel() << " height=" << inputs[0]->height() << " width=" << inputs[0]->width();
//    LOG(INFO) << "num=" << param.weights->num() << " channel=" << param.weights->channel() << " height=" << param.weights->height() << " width=" << param.weights->width();
    if (inDtype == AK_FLOAT) {
        cl_int errNum = 0;
        //LOG(INFO) << "device id= " << device << " conext = " << context;
        
        //To get the commpute command queue
        AMD_API::stream_t cm = this->_ctx->get_compute_stream();

        //To set the argument
        cl_mem memObjects[4] = { 0, 0 , 0, 0};
        
        memObjects[0] = (cl_mem)inputs[0]->data();
        memObjects[1] = (cl_mem)param.weights->data();
        memObjects[2] = (cl_mem)param.bias->data();
        memObjects[3] = (cl_mem)outputs[0]->mutable_data();

        errNum = setKernelArgs(_kernel, memObjects[0], memObjects[1],
                               memObjects[2], memObjects[3]);
        if (errNum != CL_SUCCESS)
        {
            LOG(ERROR) << "Fail to set kernel arguments";
            return SaberInvalidValue;
        }

        //LOG(INFO) << "COMPLETE SET ARGUMENT";
        //SaberTimer<AMD> timer;
        //timer.start(this->_ctx);
        cl_event event;
        errNum = clEnqueueNDRangeKernel(cm, _kernel, 3, NULL,
                                        _globalWorkSize, _localWorkSize,
                                        0, NULL, &event);
        //timer.end(this->_ctx);
        //LOG(INFO) << "elapse: " << timer.get_best_ms();
        if (errNum != CL_SUCCESS)
        {
            LOG(ERROR) << "Fail to set execution: " << errNum;
            return SaberInvalidValue;
        }
        cl_event_list list;
        list.push_back(event);
        Env<AMD>::add_event(list);
        //LOG(INFO) << "COMPLETE EXECUTION";
    }
    return SaberSuccess;
}

template class VenderFc<AMD, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
#endif
}
} // namespace anakin
