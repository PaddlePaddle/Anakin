#include "saber/lite/funcs/saber_resize.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

#include <cmath>
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"

namespace anakin{

namespace saber{

namespace lite{


static void resize_spatial(const float* src, int w_in, int h_in, float* dst, \
                            int w_out, int h_out, float* coor_buf, std::vector<Tensor<CPU, AK_FLOAT>> rows_buf){

    int* xofs = (int*)coor_buf;
    int* yofs = xofs + w_out;

    float* alpha = (float*)yofs + h_out;
    float* beta = alpha + w_out * 2;

#ifdef USE_OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    float* rows0 = rows_buf[thread_id * 2].mutable_data();
    float* rows1 = rows_buf[thread_id * 2 + 1].mutable_data();

    int prev_sy1 = -1;
    //main loop 
    for (int dy = 0; dy < h_out; dy++ ){
        int sy = yofs[dy];

        if (sy == prev_sy1){
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src + (sy+1) * w_in;

            const float* alphap = alpha;
            float* rows1p = rows1;
            
            int dx = 0;
            for ( ; dx+1 < w_out; dx += 2 ){
                int sx = xofs[dx];
                int sxn = xofs[dx+1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

#ifdef __aarch64__

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);
                alphap += 4;
#else
                float* rows1pt = rows1;
                asm volatile(
                        "vld1.32 {d0-d1}, [%[alpha]]!       @load alpha to q0\n"
                        "vld1.32 d2, [%[s1p]]               @load s1p to d2  \n"
                        "vld1.32 d3, [%[s1np]]              @load s1np to d3   \n"

                        "vmul.f32 q2, q1, q0                @mul \n"
                        "vpadd.f32 d6, d4, d5               @pair add  \n"
                        "vst1.32 d6, [%[out]]!              @store d6 to out\n"
                        "pld [%[alpha]]                     @preload alpha\n"
                        :[alpha]"+r"(alphap), [out]"+r"(rows1pt)
                        :[s1p]"r"(S1p), [s1np]"r"(S1np)
                        :"q0","q1","q2","q3"
                    );
#endif
                
            }
            
            for ( ; dx < w_out; dx++ ){
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }
        else{
            // hresize two rows

            const float* S0 = src + sy * w_in;
            const float* S1 = src + (sy+1) * w_in;

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            
            
            int dx = 0;
            float* rows0pt = rows0;
            float* rows1pt = rows1;
            for ( ; dx+1 < w_out; dx += 2 ){
                int sx = xofs[dx];
                int sxn = xofs[dx+1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;
                
                
                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);
               
                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);
                
            
                alphap += 4;
                  
                /*
                asm volatile(
                            "pld [%[alpha]]                         \n"
                            "vld1.32 {d0-d1}, [%[alpha]]!           \n"
                            "vld1.32 d2, [%[s0p]]                   \n"
                            "vld1.32 d3, [%[s0np]]                  \n"
                            "vld1.32 d4, [%[s1p]]                   \n"
                            "vld1.32 d5, [%[s1np]]                  \n"
                
                            "vmul.f32 q3, q1, q0                    \n"
                            "vmul.f32 q4, q2, q0                    \n"

                            "vpadd.f32 d10, d6, d7                  \n"
                            "vpadd.f32 d11, d8, d9                  \n"

                            "vst1.32 d10, [%[out1]]!                \n"
                            "vst1.32 d11, [%[out2]]!                \n"

                            :[out1]"+r"(rows0pt), [out2]"+r"(rows1pt), [alpha]"+r"(alphap)
                            :[s0p]"r"(S0p), [s1p]"r"(S1p),[s0np]"r"(S0np),[s1np]"r"(S1np)
                            :"q0", "q1", "q2", "q3", "q4", "q5"
                );  
                */
                
            }

            
            for ( ; dx < w_out; dx++ ){
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0]*a0 + S0p[1]*a1;
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;

        float* Dp = dst + dy * w_out;

        int nn = w_out >> 3;
        int remain = w_out - (nn << 3);

#ifdef __aarch64__ 
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);     
        
        for (; nn>0; nn--){
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p+4);
            float32x4_t _rows1n = vld1q_f32(rows1p+4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp+4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
        
#else
        if (nn > 0){
            asm volatile(
                "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
                "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
                "1:                                                      \n"
                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows0p]]                     @preload rows0p\n"

                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows1p]]                     @preload rows1p\n"
                "subs %[loopc], #1                   @loop count minus #1\n"
                "bne 1b                              @jump to 1\n"    
                :[rows0p]"+r"(rows0p), [rows1p]"+r"(rows1p), [out]"+r"(Dp), [loopc]"+r"(nn)
                :[b0]"r"(b0), [b1]"r"(b1)
                :"q0", "q1", "q2", "q3"
            );
        }
#endif

        for ( ; remain; --remain){
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }

}


void resize(const float* in_data, int count, int h_in, int w_in, \
            float* out_data, int h_out, int w_out, float* coor_buf, std::vector<Tensor<CPU, AK_FLOAT>> &rows_buf){

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;

#pragma omp parallel for
    for(int i = 0; i < count; ++i){
        resize_spatial(in_data + i * spatial_in, w_in, h_in, \
                        out_data + i * spatial_out, w_out, h_out, coor_buf, rows_buf);
    }
}

SaberResize::SaberResize(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const ResizeParam*)param;
    this->_flag_param = true;
}

SaberResize::~SaberResize() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberResize::load_param(const ParamBase *param) {
    _param = (const ResizeParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberResize::load_param(std::istream& stream, const float* weights) {
    int width_scale, height_scale;
    stream >> width_scale >> height_scale;
    _param = new ResizeParam(width_scale, height_scale);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberResize::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load resize param first\n");
        return SaberNotInitialized;
    }
    Shape sh = inputs[0]->valid_shape();
    sh[2] = sh[2] * _param->_height_scale;
    sh[3] = sh[3] * _param-> _width_scale;
    return outputs[0]->set_shape(sh);
}

SaberStatus SaberResize::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load resize param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;

    _width_scale = _param->_width_scale;
    _height_scale = _param->_height_scale;
    
    int out_w = outputs[0]->valid_shape()[3];
    int out_h = outputs[0]->valid_shape()[2];
    int in_w = inputs[0]->valid_shape()[3];
    int in_h = inputs[0]->valid_shape()[2];

#ifdef USE_OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    //allocate space 
    Shape sh_coor(1, 1, 1, 3 * (out_w+out_h)); 
    _coor_buf.re_alloc(sh_coor);
    _rows_buf.resize(2 * num_threads);
    //allocate rows buf according to threads num
    for (int i = 0; i < _rows_buf.size(); ++i){
        _rows_buf[i].re_alloc(out_w + 1);
    }
    int* xofs = (int*)_coor_buf.mutable_data();
    int* yofs =  xofs + out_w; 
    float* alpha = (float*)(yofs) + out_h;
    float* beta = alpha + out_w * 2;
    float fx, fy;
    int sx, sy;

    //pre compute coordinate in x and y direction
    for (int dx = 0; dx < out_w; dx++){
        fx = dx * (1.0 / _width_scale);
        sx = int(fx);
        fx -= sx;

        if (sx >= in_w - 1){
            sx = in_w - 2;
            fx = 1.f;
        }
        xofs[dx] = sx;
        alpha[dx * 2    ] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }

    for (int dy = 0; dy < out_h; dy++){
        fy = dy * (1.0 / _height_scale);
        sy = int(fy);
        fy -= sy;

        if (sy >= in_h - 1){
            sy = in_h - 2;
            fy = 1.f;
        }
        yofs[dy] = sy;
        beta[dy * 2    ] = 1.f - fy;
        beta[dy * 2 + 1] = fy;
    }
    this->_flag_init = true;
    return SaberSuccess;

}


SaberStatus SaberResize::dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
                                    std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {

    if (!this->_flag_init) {
        printf("init resize first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    float* dout = outputs[0]->mutable_data();
    const float* din = (float*)inputs[0]->data();

    int out_num = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int count = out_num * out_c;
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();

    resize(din, count, in_h, in_w, dout, out_h, out_w, _coor_buf.mutable_data(), _rows_buf);


#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("resize time: %f\n", ts);
    OpTimer::add_timer("resize", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}

REGISTER_LAYER_CLASS(SaberResize);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM


