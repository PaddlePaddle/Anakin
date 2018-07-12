#include "saber/core/context.h"
#include "saber/funcs/activation.h"
#include "test_saber_func_AMD.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/amd/amd_utils.h"
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace anakin::saber;

typedef TargetWrapper<AMD> API;
typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<AMD, AK_FLOAT, NCHW> TensorDf4;
typedef TensorHf4::Dtype dtype;

const int ARRAY_SIZE = 1000;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor &t0) {

    LOG(INFO)<<name<<" valid shape is ["
             <<t0.valid_shape()[0]<<", "
             <<t0.valid_shape()[1]<<", "
             <<t0.valid_shape()[2]<<", "
             <<t0.valid_shape()[3]<<"].";

    LOG(INFO)<<name<<" real shape is ["
             <<t0.shape()[0]<<", "
             <<t0.shape()[1]<<", "
             <<t0.shape()[2]<<", "
             <<t0.shape()[3]<<"].";

    LOG(INFO)<<name<<" offset is ["
             <<t0.offset()[0]<<", "
             <<t0.offset()[1]<<", "
             <<t0.offset()[2]<<", "
             <<t0.offset()[3]<<"].";
}

template<typename T>
void print_result(T *result, size_t size = ARRAY_SIZE) {
    std::string tmp;
    for (int i = 0; i < size; i++) {
        tmp.append(std::to_string(*result) + std::string(" "));
        result++;
    }
    LOG(INFO) << tmp;
}

bool CreateCLMem(int id, cl_mem mems[3],
                      float *a, float *b)
{
    ClMem mem1[3];

    API::mem_alloc(&mem1[0], sizeof(float) * ARRAY_SIZE);
    API::mem_alloc(&mem1[1], sizeof(float) * ARRAY_SIZE);
    API::mem_alloc(&mem1[2], sizeof(float) * ARRAY_SIZE);

    API::sync_memcpy(mem1[0], id, (void *) a, 0, sizeof(float) * ARRAY_SIZE, __HtoD());
    API::sync_memcpy(mem1[1], id, (void *) b, 0, sizeof(float) * ARRAY_SIZE, __HtoD());

    if (mems[0] == NULL || mems[1] == NULL || mems[2] == NULL)
    {
        LOG(ERROR) << "Failed to create memory objects." ;
        return false;
    }

    mems[0] = mem1[0].dmem;
    mems[1] = mem1[1].dmem;
    mems[2] = mem1[2].dmem;

    return true;
}

bool CreateCLMem_Async(int id, cl_mem mems[3], cl_command_queue cm,
                      float *a, float *b)
{
    ClMem mem1[3];

    API::mem_alloc(&mem1[0], sizeof(float) * ARRAY_SIZE);
    API::mem_alloc(&mem1[1], sizeof(float) * ARRAY_SIZE);
    API::mem_alloc(&mem1[2], sizeof(float) * ARRAY_SIZE);

    API::async_memcpy(mem1[0], id, (void *) a, 0, sizeof(float) * ARRAY_SIZE, cm, __HtoD());
    API::async_memcpy(mem1[1], id, (void *) b, 0, sizeof(float) * ARRAY_SIZE, cm, __HtoD());

    if (mems[0] == NULL || mems[1] == NULL || mems[2] == NULL)
    {
        LOG(ERROR) << "Failed to create memory objects." ;
        return false;
    }

    mems[0] = mem1[0].dmem;
    mems[1] = mem1[1].dmem;
    mems[2] = mem1[2].dmem;

    return true;
}

void CleanCL(cl_context context, cl_command_queue cm,
             cl_program program, cl_kernel kernel, cl_mem mems[3])
{
    if(mems != NULL)
        for (int i = 0; i < 3; i++)
        {
            if (mems[i] != 0) {
                API::mem_free((void*) mems[i]);
                LOG(INFO) << "    release mem #" << i ;
                mems[i] = 0;
            }
        }

    LOG(INFO) << "release mem done ";

    if (cm != 0)
        clReleaseCommandQueue(cm);

    LOG(INFO) << "release command done " ;

    if (kernel != 0)
        clReleaseKernel(kernel);

    LOG(INFO) << "release kernel done " ; 

    if (program != 0)
        clReleaseProgram(program);

    LOG(INFO) << "release program done " ;

    if (context != 0)
        clReleaseContext(context);
    LOG(INFO) << "release context done " ;
}

template<typename Dtype>
void memset(Dtype pattern) {

    Device<AMD> dev = Env<AMD>::cur_env()[0];

    cl_device_id id = dev.get_device();
    cl_context context = dev.get_context();
    cl_command_queue cm = dev._data_stream[0];

    size_t size = sizeof(Dtype) * ARRAY_SIZE;

    Dtype data[ARRAY_SIZE];

    for(int i = 0 ; i < ARRAY_SIZE ; i++) {
        data[i] = pattern;
    }

    cl_mem buf;
    ClMem membuf;
    API::mem_alloc(&membuf, size);
    API::mem_set(membuf, pattern, size);

    buf = membuf.dmem;

    cl_int err;
    void *data_res = clEnqueueMapBuffer(cm, buf, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
    CHECK(err == CL_SUCCESS);  

    clFlush(cm);
   
    err = CL_INVALID_VALUE;
    if(memcmp(&data[0], data_res, size) == 0){
        err = CL_SUCCESS;
    }

    Dtype *res = (Dtype *)data_res;
    print_result(res);

    clEnqueueUnmapMemObject(cm, buf, data_res, 0, NULL, NULL);
    clFinish(cm);

    CHECK(err==CL_SUCCESS);

}

/*
TEST(TestSaberFuncAMD, test_memset_f) {
   memset(2.0f); 
}
*/

TEST(TestSaberFuncAMD, test_memset) {
   memset(2); 
}

TEST(TestSaberFuncAMD, test_memcpy_h2d) {
    Device<AMD> dev = Env<AMD>::cur_env()[0];

    cl_device_id id = dev.get_device();
    cl_context context = dev.get_context();
    cl_command_queue cm = dev._data_stream[0];

    float data[ARRAY_SIZE];
    size_t size = sizeof(float) * ARRAY_SIZE;

    for(int i = 0 ; i < ARRAY_SIZE ; i++) {
        data[i] = i;
    }
    cl_mem buf;
    ClMem membuf;

    API::mem_alloc(&membuf, size);
    API::sync_memcpy(membuf, 0, (void*) &data[0], 0, size, __HtoD());

    buf = membuf.dmem;

    cl_int err;
    void *data_res = clEnqueueMapBuffer(cm, buf, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
    CHECK(err == CL_SUCCESS);
    

    clFlush(cm);
   
    err = CL_INVALID_VALUE;
    if(memcmp(&data[0], data_res, size) == 0){
        err = CL_SUCCESS;
    }

    float *res = (float *)data_res;
    print_result(res);

    clEnqueueUnmapMemObject(cm, buf, data_res, 0, NULL, NULL);
    clFinish(cm);

    CHECK(err==CL_SUCCESS);
 
}

TEST(TestSaberFuncAMD, test_memcpy_d2h) {

    Device<AMD> dev = Env<AMD>::cur_env()[0];

    cl_device_id id = dev.get_device();
    cl_context context = dev.get_context();
    cl_command_queue cm = dev._data_stream[0];

    float data[ARRAY_SIZE];
    size_t size = sizeof(float) * ARRAY_SIZE;

    for(int i = 0 ; i < ARRAY_SIZE ; i++) {
        data[i] = i;
    }
    cl_mem buf;
    ClMem membuf;
    API::mem_alloc(&membuf, size);
    API::sync_memcpy(membuf, 0, (void*) &data[0], 0, size, __HtoD());

    float data_res[ARRAY_SIZE];
    API::sync_memcpy((void *) &data_res[0], 0, membuf, 0, size, __DtoH());

   
    cl_int err = CL_INVALID_VALUE;
    if(memcmp(&data[0], &data_res[0], size) == 0){
        err = CL_SUCCESS;
    }

    float *res = (float *)data_res;
    print_result(res);
    LOG(INFO) ;
    CHECK(err==CL_SUCCESS);
 
}


TEST(TestSaberFuncAMD, test_memcpy_d2d) {

    Device<AMD> dev = Env<AMD>::cur_env()[0];

    cl_device_id id = dev.get_device();
    cl_context context = dev.get_context();
    cl_command_queue cm = dev._data_stream[0];

    float data[ARRAY_SIZE];
    size_t size = sizeof(float) * ARRAY_SIZE;

    for(int i = 0 ; i < ARRAY_SIZE ; i++) {
        data[i] = i;
    }
    cl_mem buf, buf2;
    ClMem membuf, membuf2;
    API::mem_alloc(&membuf, size);
    API::sync_memcpy(membuf, 0, (void*) &data[0], 0, size, __HtoD());

    API::mem_alloc(&membuf2, size);
    API::sync_memcpy(membuf2, 0, membuf, 0, size, __DtoD());

    buf = membuf.dmem;
    buf2 = membuf2.dmem;
   
    cl_int err;
    void *data_res = clEnqueueMapBuffer(cm, buf2, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
    CHECK(err == CL_SUCCESS);
    clFlush(cm);
   
    err = CL_INVALID_VALUE;
    if(memcmp(&data[0], data_res, size) == 0){
        err = CL_SUCCESS;
    }

    float *res = (float *)data_res;
    print_result(res);

    clEnqueueUnmapMemObject(cm, buf2, data_res, 0, NULL, NULL);
    clFinish(cm);

    CHECK(err==CL_SUCCESS);
 
}

#if 1
#define enable_timer 1
TEST(TestSaberFuncAMD, test_func_tensor) {

    TensorDf4 tdev1;
    TensorDf4 tdev2;
    TensorDf4 tdev3;
    TensorHf4 thost1;
    TensorHf4 thost2;
    TensorHf4 thost3;


    Shape sh0 (4,4,4,4);

    tdev1.re_alloc(sh0);
    tdev2.re_alloc(sh0);
    tdev3.re_alloc(sh0);
    thost1.re_alloc(sh0);
    thost2.re_alloc(sh0);
    thost3.re_alloc(sh0);

    
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float *a = thost1.mutable_data();
    float *b = thost2.mutable_data();
    for (int i = 0; i < sh0.count(); i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    tdev1.copy_from(thost1);
    tdev2.copy_from(thost2);

    cl_context context = 0;
    cl_command_queue cm = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_int errNum;


    Device<AMD> dev = Env<AMD>::cur_env()[0];


    Context<AMD> ctx(AMD_API::get_device_id(),0,0);
    device = dev.get_device();
    // Create an OpenCL context on first available platform
    context = dev.get_context();

    LOG(INFO) << "device id= " << device << " conext = " << context;
    if (context == NULL)
    {
        LOG(ERROR) << "Failed to create OCL context." ;
        //return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    cm = ctx.get_compute_stream();
    if (cm == NULL)
    {
        CleanCL(NULL, NULL, program, kernel, NULL);
        //return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateCLProgram(context, device, "amd_test.cl");
    if (program == NULL)
    {
        CleanCL(NULL, NULL, program, kernel, NULL);
        //return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "plus_kernel", NULL);
    if (kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel" ;
        CleanCL(NULL, NULL, program, kernel, NULL);
        //return 1;
    }

    ClMem mem1[3] = {tdev1.mutable_data(), tdev2.mutable_data(), tdev3.mutable_data()};

    cl_mem mems[3] = {mem1[0].dmem, mem1[1].dmem, mem1[2].dmem};//{(cl_mem) tdev1.mutable_data(), (cl_mem) tdev2.mutable_data(), (cl_mem) tdev3.mutable_data()};
    
    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem),  & mems[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), & mems[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), & mems[2]);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Failed to set kernel arguments." ;
        CleanCL(NULL, NULL, program, kernel, NULL);
        //return 1;
    }

    size_t g_work_size[1] = {(size_t)sh0.count()};
    size_t l_work_size[1] = { 1 };

#if enable_timer
    SaberTimer<AMD> timer;
    timer.start(ctx);

    cl_event ue = clCreateUserEvent(context, NULL);
    cl_event event;
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(cm, kernel, 1, NULL,
                                    g_work_size, l_work_size,
                                    1, &ue, &event);
#else

    cl_event event;
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(cm, kernel, 1, NULL,
                                    g_work_size, l_work_size,
                                    0, NULL, &event);
#endif
     
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Failed to queue kernel for execution." ;
        CleanCL(NULL, NULL, program, kernel, NULL);
        //return 1;
    }


#if enable_timer
    sleep(2);
    clSetUserEventStatus(ue, CL_COMPLETE);
    AMD_API::destroy_event(ue);
    AMD_API::sync_event(event);
    timer.end(ctx);
    LOG(INFO) << "Executing time "<< timer.get_average_ms() << " ms";
#else
    AMD_API::sync_event(event);
#endif

    thost3.copy_from(tdev3);
    dtype *result = thost3.mutable_data();

    print_result(result, sh0.count());
    
    CleanCL(NULL, NULL, program, kernel, NULL);
    LOG(INFO) << "Executed program succesfully." ;

}
#endif

#if 1
TEST(TestSaberFuncAMD, test_saber_impl_amd) {
    cl_context context = 0;
    cl_command_queue cm = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem mems[3] = { 0, 0, 0 };
    cl_int errNum;


    Device<AMD> dev = Env<AMD>::cur_env()[0];


    SaberTimer<AMD> timer;


    Context<AMD> ctx(AMD_API::get_device_id(),0,0);
    device = dev.get_device();
    context = dev.get_context();

    LOG(INFO) << "device id= " << device << " conext = " << context;
    if (context == NULL)
    {
        LOG(ERROR) << "Failed to create OCL context." ;
        //return 1;
    }

    cm = ctx.get_compute_stream();
    if (cm == NULL)
    {
        LOG(INFO) << "clean cl";
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    LOG(INFO) << "create cl test kernel";
    program = CreateCLProgram(context, device, "amd_test.cl");
    if (program == NULL)
    {

        LOG(INFO) << "clean cl";
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    LOG(INFO) << "create cl plus kernel";
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "plus_kernel", NULL);
    if (kernel == NULL)
    {
        LOG(ERROR) << "Failed to create kernel" ;
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    //if (!CreateCLMem(0, mems, a, b))
    if (!CreateCLMem_Async(0, mems, cm, a, b))
    {
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    LOG(INFO) << "set cl args";
    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mems[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mems[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mems[2]);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Failed to set kernel arguments." ;
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    timer.start(ctx);

    size_t g_work_size[1] = {ARRAY_SIZE };
    size_t l_work_size[1] = { 1 };

    cl_event ue = clCreateUserEvent(context, NULL);
    cl_event event;
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(cm, kernel, 1, NULL,
                                    g_work_size, l_work_size,
                                    1, &ue, &event);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Failed to queue kernel for execution." ;
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }

    sleep(2);
    clSetUserEventStatus(ue, CL_COMPLETE);

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(cm, mems[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 1, &event, NULL);
    if (errNum != CL_SUCCESS)
    {
        LOG(ERROR) << "Error reading result buffer." ;
        CleanCL(NULL, NULL, program, kernel, mems);
        //return 1;
    }



    clFlush(ctx.get_compute_stream());
    clFinish(ctx.get_compute_stream());
    timer.end(ctx);

    LOG(INFO) << "Executing time "<< timer.get_average_ms() << " ms";

    print_result(result);

    LOG(INFO) << "Executed program succesfully." ;
    CleanCL(NULL, NULL, program, kernel, mems);
//    return 0;

}
#endif 
int main(int argc, const char** argv){
    Env<AMD>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

