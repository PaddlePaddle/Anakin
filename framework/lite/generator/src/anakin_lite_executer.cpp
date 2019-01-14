#include "saber/saber_types.h"
#include "framework/lite/code_gen_cpp.h"
#include "framework/core/types.h"

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::lite;

void anakin_lite_executer(const char* model_name, const char* model_path, const char* precision_path, \
    const char* calibrator_path, const int lite_mode, const char* output_path, const bool flag_aot, \
    const bool debug_mode = false, const int batch_size = 1) {
    // constructs
    GenCPP<X86, Precision::FP32> code_gen(model_name, output_path, precision_path, calibrator_path,\
                                            lite_mode, flag_aot);
    if (!code_gen.extract_graph(model_path, batch_size)) {
        LOG(ERROR) << "extract error on : " << model_path;
    }
    // gen
    code_gen.gen_files(debug_mode);
}


int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    if (argc < 5) {
        LOG(ERROR) << "Some arguments not supplied!";
        LOG(ERROR) << "usage: " << argv[0] << " model_name model_weights_path(xxx.c.bin) output_path aot_mode debug_mode batch_size  lite_mode precision_path calibrator_path";
        LOG(ERROR) << "model_name: output lib and api name";
        LOG(ERROR) << "model_weights_path: path to your anakin model";
        LOG(ERROR) << "output_path: output path";
        LOG(ERROR) << "aot_mode: >0: aot mode, generate .h and .cpp; 0: general mode, generate .lite.info and .lite.bin";
        LOG(ERROR) << "debug_mode: debug mode, only for aot mode, 0:no debug info, 1:with debug info";
        LOG(ERROR) << "batch_size: default 1";
        LOG(ERROR) << "lite_mode: generate lite model, default 0";
        LOG(ERROR) << "precision_path: precision file path";
        LOG(ERROR) << "calibrator_path: calirator file path";

        return 1;
    }
    const char* model_name = argv[1];
    const char* model_path = argv[2];
    const char* output_path = argv[3];
    bool flag_aot = atoi(argv[4]) > 0;
    bool flag_debug = false;
    if (argc > 5) {
        flag_debug = atoi(argv[5]) > 0;
    }
    int batch_size = 1;
    if (argc > 6){
        batch_size = atoi(argv[6]);
    }
    int lite_mode = 0;
    if (argc > 7){
        lite_mode = atoi(argv[7]);
    }
    const char* precision_path = "";
    if (argc > 8){
        precision_path = argv[8];
    }
    const char* calibrator_path = "";
    if (argc > 9){
        calibrator_path = argv[9];
    }
    anakin_lite_executer(model_name, model_path, precision_path, calibrator_path, lite_mode, \
                         output_path, flag_aot, flag_debug, batch_size);
    return 0;
}
