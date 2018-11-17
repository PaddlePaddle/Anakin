#include "test_lite.h"
//!change here according to your own model
//#include "mobilenet.h"
#include <fstream>

using namespace anakin::saber;
using namespace anakin::saber::lite;
typedef Tensor<CPU> TensorHf;

std::string model_file_name;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 1;
int FLAGS_epoch = 1;
int FLAGS_threads = 1;
int FLAGS_cluster = 0;

TEST(TestSaberLite, test_lite_model) {

    //! create runtime context
    LOG(INFO) << "create runtime context";
    Context* ctx1 = new Context;
    ctx1->set_run_mode((PowerMode)FLAGS_cluster, FLAGS_threads);
    //! test threads
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    //! change here according to your own model
    //bool load_flag = mobilenet_load_param(model_file_name.c_str());
    //CHECK_EQ(load_flag, true) << "load model: " << model_file_name << " failed";
    LOG(INFO) << "load model: " << model_file_name << " successed";

//! load model from memory
//    std::fstream fp(model_file_name, std::ios::in | std::ios::binary);
//    std::stringstream str_str;
//    str_str << fp.rdbuf();
//    std::string str(str_str.str());
//    LOG(INFO) << "get fstream";
//    const char* w_ptr = str.c_str();
//    bool load_flag = mobilenet_load_weights(w_ptr);
//    LOG(WARNING) << "load anakin model file from " << model_file_name << " ...";
//    CHECK_EQ(load_flag, true) << "load model: " << model_file_name << " failed";
//    LOG(INFO) << "load model: " << model_file_name << " successed";

    //! init net
    //! change here according to your own model
    //bool init_flag = mobilenet_init(*ctx1);
    //CHECK_EQ(init_flag, true) << "init failed";
    LOG(INFO) << "init successed";

    //! change here according to your own model
    std::vector<TensorHf*> vtin_mobilenet;// = mobilenet_get_in();
    LOG(INFO) << "number of input tensor: " << vtin_mobilenet.size();
    for (int i = 0; i < vtin_mobilenet.size(); ++i) {
        TensorHf* tin_mobilenet = vtin_mobilenet[i];

        //!input shape can be changed at each prediction, after reshape input, call xx_init() api;
        //tin_mobilenet->reshape(Shape(1, 3, 224, 224));

        LOG(INFO) << "input tensor size: ";
        Shape shin_mobilenet = tin_mobilenet->valid_shape();
        for (int j = 0; j < tin_mobilenet->dims(); ++j) {
            LOG(INFO) << "|---: " << shin_mobilenet[j];
        }
        //! feed data to input
        //! feed input image to input tensor
        fill_tensor_const(*tin_mobilenet, 1.f);
    }

    //! call init api after reshape input
    //mobilenet_init(*ctx1);

    //! change here according to your own model
    std::vector<TensorHf*> vtout_mobilenet;// = mobilenet_get_out();
    LOG(INFO) << "number of output tensor: " << vtout_mobilenet.size();
    for (int i = 0; i < vtout_mobilenet.size(); i++) {
        TensorHf* tout = vtout_mobilenet[i];
        LOG(INFO) << "output tensor size: ";
        Shape shout = tout->valid_shape();
        for (int j = 0; j < tout->dims(); ++j) {
            LOG(INFO) << "|---: " << shout[j];
        }
    }

    SaberTimer my_time;
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    my_time.start();
    SaberTimer t1;
    for (int i = 0; i < FLAGS_epoch; i++) {

        for (int j = 0; j < vtin_mobilenet.size(); ++j) {
            fill_tensor_const(*vtin_mobilenet[j], 1.f);
            printf("input mean val: %.6f\n", tensor_mean(*vtin_mobilenet[j]));
        }
        t1.clear();
        t1.start();
        //! change here according to your own model
        //mobilenet_prediction();
        t1.end();
        float tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
        LOG(INFO) << "mobilenet iter: " << i << ", time: " << tdiff << "ms";
        for (int i = 0; i < vtout_mobilenet.size(); ++i) {
            double mean_val = tensor_mean(*vtout_mobilenet[i]);
            LOG(INFO) << "mobilenet output mean: " << mean_val;
        }
    }
    my_time.end();

    LOG(INFO) << model_file_name << " batch_size " << FLAGS_num << " average time " << to/ FLAGS_epoch << \
            ", min time: " << tmin << "ms, max time: " << tmax << " ms";

    for (int i = 0; i < vtout_mobilenet.size(); ++i) {
        double mean_val = tensor_mean(*vtout_mobilenet[i]);
        LOG(INFO) << "mobilenet output mean: " << mean_val;
    }


#ifdef ENABLE_OP_TIMER
    OpTimer::print_timer();
#endif //ENABLE_OP_TIMER

    //! change here according to your own model
    //mobilenet_release_resource();
    delete ctx1;
}
int main(int argc, const char** argv){

    Env::env_init();
    // initial logger
    logger::init(argv[0]);

    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <model_file> <num> <warmup_iter> <epoch>";
    LOG(INFO)<< "   model_file:     path to model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   warmup_iter:    warm up iterations default to 10";
    LOG(INFO)<< "   epoch:          time statistic epoch default to 10";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores";
    LOG(INFO)<< "   threads:        set openmp threads";
    if (argc < 2) {
        LOG(ERROR) << "You should fill in the variable model_dir and model_file at least.";
        return 0;
    }
    if (argc > 1) {
        model_file_name = argv[1];
    }

    if (argc > 2) {
        FLAGS_num = atoi(argv[2]);
    }
    if (argc > 3) {
        FLAGS_warmup_iter = atoi(argv[3]);
    }
    if (argc > 4) {
        FLAGS_epoch = atoi(argv[4]);
    }
    if (argc > 5) {
        FLAGS_cluster = atoi(argv[5]);
        if (FLAGS_cluster < 0) {
            FLAGS_cluster = 0;
        }
        if (FLAGS_cluster > 1) {
            FLAGS_cluster = 1;
        }
    }
    if (argc > 6) {
        FLAGS_threads = atoi(argv[6]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}
