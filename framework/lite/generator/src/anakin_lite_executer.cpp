#include "saber/saber_types.h"
#include "framework/lite/code_gen_cpp.h"
#include "framework/core/types.h"

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::lite;

void anakin_lite_executer(const char* model_name, const char* model_path, const char* output_path = "./") {
    // constructs 
	GenCPP<X86, AK_FLOAT, Precision::FP32> code_gen(model_name, output_path);
	if(! code_gen.extract_graph(model_path)) {
		LOG(ERROR) << "extract error on : " << model_path;
	}
	// gen
	code_gen.gen_files();
}


int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	if(argc < 3) {
		LOG(ERROR) << "Some arguments not supplied!";
		LOG(ERROR) << "usage: " << argv[0] << " model_name model_weights_path(xxx.anakin.bin) output_path";
		LOG(ERROR) << "model_name: output lib and api name";
		LOG(ERROR) << "model_weights_path: path to your anakin model";
		LOG(ERROR) << "output_path: output path";
		return 1;
	}
	const char* model_name = argv[1];
	const char* model_path = argv[2];
	if(argc == 3) {
		anakin_lite_executer(model_name, model_path);
	} else { // > 3
		const char* output_path = argv[3];
		anakin_lite_executer(model_name, model_path, output_path);
	}
	return 0;
}
