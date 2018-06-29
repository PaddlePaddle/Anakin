#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_SCALE_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_SCALE_H

#include "saber/funcs/impl/impl_scale.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderScale<BM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>,
        Tensor<BM, outDtype, LayOutType_out>,
        Tensor<BM, OpDtype, LayOutType_op>,
        ScaleParam<Tensor<BM, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderScale() {}

    ~VenderScale() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ScaleParam<OpTensor>& param, Context<BM>& ctx) {

        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ScaleParam<OpTensor>& param, Context<BM> &ctx) {

    }
    
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ScaleParam<OpTensor>& param) {

        const InDataType in_data = *(inputs[0]->data());
        OutDataType out_data = *(outputs[0]->mutable_data());

        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();

        int axis = (param.num_axes == 0) ? 0 : param.axis;
        int num_axes = param.num_axes >=0 ? param.num_axes : inputs[0]->shape().dims() - axis;

        int outer_dim = inputs[0]->count(0, axis);
        int inner_dim = inputs[0]->count(axis + num_axes, inputs[0]->shape().dims());
        int scale_dim = inputs[0]->count(axis, axis + num_axes);
        /* if (inputs.size() == 1) { */
        /*     CHECK_EQ(scale_dim, param.scale_w.size()) << "scale dim not valid"; */
        /* } */

        float* scale_data = &param.scale_w[0];
        bm_device_mem_t* data_extension = new bm_device_mem_t();
        int size = input_n * input_c * input_h * input_w;
        bm_malloc_device_byte(_handle, data_extension, size * sizeof(float));
        BMDNN_CHECK(bmdnn_scale_forward(_handle, in_data, bm_mem_from_system(scale_data),
                input_n, input_c, input_h, input_w,
                scale_dim, inner_dim, 0,
                *data_extension, out_data));
        
        if (param.bias_term) {
            float* host_bias = &param.scale_b[0];
            float* host_extension = new float[size];
            int dim = inner_dim * scale_dim;
            for (int i = 0; i < size; ++i) {
                 int bias_dim = (i % dim) / inner_dim;
                 host_extension[i] = host_bias[bias_dim];
            }

            bm_flush(get_bm_handle());
            BMDNN_CHECK(bmdnn_bias_forward(_handle, out_data, bm_mem_from_system(host_extension),
                    outer_dim, scale_dim * inner_dim, out_data));

            delete [] host_bias;
            delete [] host_extension;
        }
        bm_free_device(_handle, *data_extension);
        return SaberSuccess;
    }
private:
    bm_handle_t _handle;
};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_SCALE_H
