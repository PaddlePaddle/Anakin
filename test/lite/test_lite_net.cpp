#include "test_lite.h"
#include "saber/lite/net/net_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;
typedef Tensor<CPU, AK_FLOAT> TensorHf;

std::string info_file;
std::string weights_file;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 1;
int FLAGS_epoch = 1;
int FLAGS_threads = 1;
int FLAGS_cluster = 0;

TEST(TestSaberLite, test_lite_model) {

    //! create net, with power mode and threads
    Net net((PowerMode)FLAGS_cluster, FLAGS_threads);
    //! you can also set net param according to your device
    //net.set_run_mode((PowerMode)FLAGS_cluster, FLAGS_threads);
    //net.set_device_cache(32000, 2000000);
    //! load model
    SaberStatus flag = net.load_model(info_file.c_str(), weights_file.c_str());
    CHECK_EQ(flag, SaberSuccess) << "load model: " << info_file << ", " << weights_file << " failed";
    LOG(INFO) << "load model: " << info_file << ", " << weights_file << " successed";
}
int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);

    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <info_file> <weights_file> <num> <warmup_iter> <epoch>";
    LOG(INFO)<< "   info_file:      path to model info";
    LOG(INFO)<< "   weights_file:   path to model weights";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   warmup_iter:    warm up iterations default to 10";
    LOG(INFO)<< "   epoch:          time statistic epoch default to 10";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores";
    LOG(INFO)<< "   threads:        set openmp threads";
    if(argc < 3) {
        LOG(ERROR) << "You should fill in the variable model_dir and model_file at least.";
        return 0;
    }
    info_file = argv[1];
    weights_file = argv[2];

    if(argc > 3) {
        FLAGS_num = atoi(argv[3]);
    }
    if(argc > 4) {
        FLAGS_warmup_iter = atoi(argv[4]);
    }
    if(argc > 5) {
        FLAGS_epoch = atoi(argv[5]);
    }
    if(argc > 6) {
        FLAGS_cluster = atoi(argv[6]);
        if (FLAGS_cluster < 0) {
            FLAGS_cluster = 0;
        }
        if (FLAGS_cluster > 1) {
            FLAGS_cluster = 1;
        }
    }
    if(argc > 7) {
        FLAGS_threads = atoi(argv[7]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}
