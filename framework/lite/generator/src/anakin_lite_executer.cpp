#include "saber/saber_types.h"
#include "framework/lite/code_gen_cpp.h"
#include "framework/core/types.h"

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::lite;

void anakin_lite_executer(const char* model_name, const char* model_path, const char* output_path, \
	const bool flag_aot, const bool debug_mode = false) {
    // constructs 
	GenCPP<X86, AK_FLOAT, Precision::FP32> code_gen(model_name, output_path, flag_aot);
	if(! code_gen.extract_graph(model_path)) {
		LOG(ERROR) << "extract error on : " << model_path;
	}
	// gen
	code_gen.gen_files(debug_mode);
}


int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	if(argc < 5) {
		LOG(ERROR) << "Some arguments not supplied!";
		LOG(ERROR) << "usage: " << argv[0] << " model_name model_weights_path(xxx.anakin.bin) output_path aot_mode debug_mode";
		LOG(ERROR) << "model_name: output lib and api name";
		LOG(ERROR) << "model_weights_path: path to your anakin model";
		LOG(ERROR) << "output_path: output path";
        LOG(ERROR) << "aot_mode: >0: aot mode, generate .h and .cpp; 0: general mode, generate .lite.info and .lite.bin";
        LOG(ERROR) << "debug_mode: debug mode, only for aot mode, 0:no debug info, 1:with debug info";
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
    anakin_lite_executer(model_name, model_path, output_path, flag_aot, flag_debug);
	return 0;
}
