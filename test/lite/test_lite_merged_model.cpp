#include "test_lite.h"
#include "saber/lite/net/net_lite.h"
#include "saber/lite/net/saber_factory_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;
typedef Tensor<CPU> TensorHf;

std::string lite_model;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 1;
int FLAGS_epoch = 1;
int FLAGS_threads = 1;
int FLAGS_cluster = 0;
bool FLAGS_set_archs = false;
ARMArch FLAGS_arch = A73;

TEST(TestSaberLite, test_lite_model) {
    //! create net, with power mode and threads
    Net net((PowerMode)FLAGS_cluster, FLAGS_threads);
    //! you can also set net param according to your device
    net.set_run_mode((PowerMode)FLAGS_cluster, FLAGS_threads);
    if (FLAGS_set_archs) {
        net.set_device_arch(FLAGS_arch);
        LOG(INFO) << "arm arc: " << FLAGS_arch;
    }
    net.set_device_cache(32 * 1024, 512* 1024);
    //! load merged model
    SaberStatus flag = net.load_model(lite_model.c_str());
    CHECK_EQ(flag, SaberSuccess) << "load model: " << lite_model << " failed";
    LOG(INFO) << "load model: " << lite_model << " successed";

    std::vector<TensorHf*> vtin = net.get_input();
    LOG(INFO) << "number of input tensor: " << vtin.size();
    for (int i = 0; i < vtin.size(); ++i) {
        TensorHf* tin = vtin[i];
        //! reshape input before prediction
        Shape shin = tin->valid_shape();
        shin[0] = FLAGS_num;
        tin->reshape(shin);
        //tin->reshape(Shape(1, 3, 224, 224));
        LOG(INFO) << "input tensor size: ";
        //Shape shin = tin->valid_shape();
        for (int j = 0; j < tin->dims(); ++j) {
            LOG(INFO) << "|---: " << shin[j];
        }
        //! feed data to input
        //! feed input image to input tensor
        fill_tensor_const(*tin, 1.f);
    }

    //! change here according to your own model
    std::vector<TensorHf*> vtout = net.get_output();
    LOG(INFO) << "number of output tensor: " << vtout.size();
    for (int i = 0; i < vtout.size(); i++) {
        TensorHf* tout = vtout[i];
        LOG(INFO) << "output tensor size: ";
        Shape shout = tout->valid_shape();
        for (int j = 0; j < tout->dims(); ++j) {
            LOG(INFO) << "|---: " << shout[j];
        }
    }

    for (int i = 0; i < FLAGS_warmup_iter; ++i) {
        for (int i = 0; i < vtin.size(); ++i) {
            fill_tensor_const(*vtin[i], 1.f);
        }
        net.prediction();
    }
    SaberTimer my_time;
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    my_time.start();
    SaberTimer t1;
    for (int i = 0; i < FLAGS_epoch; ++i) {
        for (int i = 0; i < vtin.size(); ++i) {
            fill_tensor_const(*vtin[i], 1.f);
        }
        t1.clear();
        t1.start();
        net.prediction();
        t1.end();
        float tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
        LOG(INFO) << "iter: " << i << ", time: " << tdiff << "ms";
    }
    for (int i = 0; i < vtout.size(); ++i) {
#ifdef ENABLE_DEBUG
        const float* ptr = vtout[i]->data();
        for (int j = 0; j < vtout[i]->valid_size(); ++j) {
            printf("%f ", ptr[j]);
            if ((j + 1) % 10 == 0) {
                printf("\n");
            }
        }
        printf("\n");
#endif
        double mean_val = tensor_mean(*vtout[i]);
        LOG(INFO) << "output mean: " << mean_val;
    }
    my_time.end();
    LOG(INFO) << lite_model << " batch_size " << FLAGS_num << " average time " << to / FLAGS_epoch << \
            ", min time: " << tmin << "ms, max time: " << tmax << " ms";
#ifdef ENABLE_OP_TIMER
    OpTimer::print_timer();
#endif //ENABLE_OP_TIMER
}
int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);

    Env::env_init();

    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <lite model> <num> <warmup_iter> <epoch>";
    LOG(INFO)<< "   lite_model:     path to anakin lite model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   warmup_iter:    warm up iterations default to 10";
    LOG(INFO)<< "   epoch:          time statistic epoch default to 10";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores, 2: all cores, 3: threads not bind to specify cores";
    LOG(INFO)<< "   threads:        set openmp threads";
    if(argc < 2) {
        LOG(ERROR) << "You should fill in the variable lite model at least.";
        return 0;
    }
    lite_model = argv[1];

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
        if (FLAGS_cluster > 5) {
            FLAGS_cluster = 5;
        }
    }
    if (argc > 6) {
        FLAGS_threads = atoi(argv[6]);
    }
    if (argc > 7) {
        FLAGS_set_archs = true;
        if (atoi(argv[7]) > 0) {
            FLAGS_arch = (ARMArch)atoi(argv[7]);
        } else {
            FLAGS_arch = ARM_UNKOWN;
        }
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}
