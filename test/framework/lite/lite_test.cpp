#include "lite_test.h"

#ifdef USE_CUDA
using Target = NV;
#endif
#ifdef USE_X86_PLACE
using Target = X86;
#endif
#ifdef USE_ARM_PLACE
using Target = ARM;
#endif

// resnet 50
std::string model_path = "/home/cuichaowen/anakin2/anakin2/benchmark/CNN/models/Resnet50.anakin.bin";

TEST(AnakinLiteTest, net_execute_base_test) {
    // constructs 
	GenCPP code_gen("Resnet50", "/home/cuichaowen/github_anakin/Anakin/build");
	if(! code_gen.extract_graph(model_path)) {
		LOG(ERROR) << "extract error on : " << model_path;
	}

	// gen
	code_gen.gen_files();


    // save the optimized model to disk.
    /*std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }*/
}


int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
