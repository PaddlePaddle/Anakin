#include "core/context.h"
#include "funcs/concat.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;


template <typename dtype, typename TargetType_D, typename TargetType_H>
void concat_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                  std::vector<Tensor<TargetType_H>*>& outputs,
                  ConcatParam<TargetType_D>& param) {
    int axis = param.axis;
    int num = outputs[0]->num();
    int channel = outputs[0]->channel();
    int height = outputs[0]->height();
    int width = outputs[0]->width();

    Shape out_sh = outputs[0]->valid_shape();
    int out_concat_axis = out_sh[axis];
    int num_concats = inputs[0]->count_valid(0, param.axis);
    int concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    dtype* dout = (dtype*)outputs[0]->mutable_data();

    int total_size = out_concat_axis * concat_input_size;

    for (int k = 0; k < num_concats; k++) {
        dtype* dout_ptr = dout + k * total_size;
        int out_size = 0;

        for (int i = 0; i < inputs.size(); i++) {
            Shape in_sh = inputs[i]->valid_shape();
            int size = in_sh[axis] * concat_input_size;
            const dtype* din = (dtype*)inputs[i]->data();
            const dtype* din_ptr = din + k * size;
            dtype* dout_ptr_axis = dout_ptr + out_size;

            for (int j = 0; j < size; j++) {
                dout_ptr_axis[j] = din_ptr[j];
            }

            out_size += size;
        }
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    int num = 1;
    int channel = 16;
    int height = 4;
    int width = 4;
    int axis1 = 3;
    TestSaberBase<TargetType_D, TargetType_H, Dtype, Concat, ConcatParam> testbase(2, 1);
    Shape input_shape({num, height, width, channel}, Layout_NHWC);
    Shape input_shape2({1, 4, 4, 16}, Layout_NHWC);

    for (auto shape : {
                input_shape, input_shape2
            }) {
        for (auto axis : {
                    axis1
                }) {
            ConcatParam<TargetType_D> param(axis);
            testbase.set_param(param);
            std::vector<Shape> shape_v;
            shape_v.push_back(shape);
            Shape shin = shape;
            shin[axis] = 2;
            shape_v.push_back(shin);
            Shape shin2 = shape;
            shin2[axis] = 4;
            shape_v.push_back(shin2);
            testbase.set_input_datatype(AK_UINT8);
            testbase.set_input_shape(shape_v);
            testbase.run_test(concat_basic<unsigned char, TargetType_D, TargetType_H>);
        }
    }
}

TEST(TestSaberFunc, test_func_concat) {
#ifdef USE_X86_PLACE
    if(jit::mayiuse(jit::avx512_core)){
        test_model<AK_INT8, X86, X86>();
    }
#endif
}


int main(int argc, const char** argv) {

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

