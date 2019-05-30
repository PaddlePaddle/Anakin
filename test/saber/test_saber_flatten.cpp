#include "saber/core/context.h"
#include "saber/funcs/flatten.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

/**
 * @brief naive flatten: change input tensor's shape[NCHW] to N*(CHW)*1*1.
 *
 * @tparam dtype
 * @tparam TargetType_D
 * @tparam TargetType_H
 * @param input
 * @param output
 * @param param
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void flatten_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, FlattenParam<TargetType_D>& param) {

    CHECK_EQ(input[0]->dims(), 4) << "flatten only support 4d(NCHW) layout";
    //flattening
    Shape shape_out;
    shape_out.resize(2);
    shape_out.set_layout(Layout_NW);
    shape_out[0] = input[0]->num();
    shape_out[1] = input[0]->valid_size() / input[0]->num();
    output[0]->set_shape(shape_out);

    // mlu realize flatten by reshape, and copy the res back to host
	// This test only check the shape ,but not the res.
	// in addition, mlu only support 4d tensor.
	// so we need the following code.
	if (std::is_same<TargetType_D, MLU>::value) {
		int size = input[0] -> valid_size();
		dtype* in_data = (dtype*)(input[0]->mutable_data());
		dtype* out_data = (dtype*)(output[0]->mutable_data());
		for (int i = 0; i < size; ++i) {
			out_data[i] = in_data[i];
		}
		Shape tmp_out({shape_out[0], shape_out[1], 1, 1});
		output[0]->set_shape(tmp_out);
	}
}

TEST(TestSaberFunc, test_op_flatten) {

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Flatten, FlattenParam> testbase;

    for (int w_in : {
                2, 8, 16
            }) {
        for (int h_in : {
                    2, 8, 32
                }) {
            for (int ch_in : {
                        2, 3, 8, 64
                    }) {
                for (int num_in : {
                            1, 21, 32
                        }) {
                    int out_num = w_in * 2;
                    DLOG(INFO) << "num_in: " << num_in;
                    DLOG(INFO) << "ch_in: " << ch_in;
                    DLOG(INFO) << "h_in: " << h_in;
                    DLOG(INFO) << "w_in: " << w_in;
                    Shape shape({num_in, ch_in, h_in, w_in});
                    FlattenParam<X86> param;
                    testbase.set_param(param);
                    testbase.set_rand_limit(1, 12);
                    testbase.set_input_shape(shape);
                    testbase.run_test(flatten_cpu_base<float, X86, X86>);
                }
            }
        }
    }
#endif

#ifdef USE_MLU
    Env<MLU>::env_init();
    TestSaberBase<MLU, MLUHX86, AK_FLOAT, Flatten, FlattenParam> testbase_mlu;
    for (int w_in : {2, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    int out_num = w_in * 2;
                    DLOG(INFO) << "num_in: " << num_in;
                    DLOG(INFO) << "ch_in: " << ch_in;
                    DLOG(INFO) << "h_in: " << h_in;
                    DLOG(INFO) << "w_in: " << w_in;
                    Shape shape({num_in, ch_in, h_in, w_in});
                    FlattenParam<MLU> param;
                    testbase_mlu.set_param(param);
                    testbase_mlu.set_rand_limit(1, 12);
                    testbase_mlu.set_input_shape(shape);
                    testbase_mlu.run_test(flatten_cpu_base<float, MLU, MLUHX86>, 0.02, true);
                }
            }
        }
    }
#endif  // USE_MLU

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
