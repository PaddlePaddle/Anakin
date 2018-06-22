#include "saber/saber_types.h"
#include "framework/lite/code_gen_cpp.h"
#include "framework/core/types.h"

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::lite;

void anakin_lite_executer(const char* model_name, const char* model_path, const char* output_path = "./") {
    // constructs 
	GenCPP<NV, AK_FLOAT, Precision::FP32> code_gen(model_name, output_path);
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
		int a = 3/0;
		LOG(ERROR) << "Some arguments not supplied!";
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
