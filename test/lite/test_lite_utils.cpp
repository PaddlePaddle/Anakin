#include "test_lite.h"
#include "saber/lite/utils/cv_utils.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 1;
int h = 1920;
int w = 720;
int ww = 112;
int hh = 288;
int angle = 90;
int flip_num = 1;
typedef Tensor<CPU> TensorHf4;

#define COMPARE_RESULT 1

void resize_uv_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, float width_scale, float height_scale) {

    const int resize_coef_bits = 11;
    const int resize_coef_scale = 1 << resize_coef_bits;
    // LOG(INFO) << "input w, h:" << w_in << ", " << h_in;
    // LOG(INFO) << "output w, h:" << w_out << ", " << h_out;

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;

    int* buf = new int[w_out * 2 + h_out * 2];
    int* xofs = buf;//new int[w];
    int* yofs = buf + w_out;//new int[h];

    float* ialpha = new float[w_out * 2];//new short[w * 2];
    float* ibeta = new float[h_out * 2];//new short[h * 2];

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;

    for (int dx = 0; dx < w_out / 2; dx++){
        fx = (float)((dx + 0.5) * width_scale - 0.5);
        sx = floor(fx);
        //printf("%.2f, %d, %d\n", fx, dx, sx);
        fx -= sx;

        if (sx < 0){
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w_in - 1){
            sx = w_in - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx);
        float a1 = fx;

        ialpha[dx * 2] = a0;
        ialpha[dx * 2 + 1] = a1;
    }

    for (int dy = 0; dy < h_out; dy++) {
        fy = (float)((dy + 0.5) * height_scale - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0){
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1){
            sy = h_in - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy);
        float b1 =        fy;

        ibeta[dy * 2] = b0;
        ibeta[dy * 2 + 1] = b1;
    }
    // for (int i = 0; i < w_out; i++)
    //     printf("%.2f ", ialpha[i]);
    // printf("\n");
    // for (int i = 0; i < h_out * 2; i++)
    //     printf("%.2f ", ibeta[i]);
    // printf("\n");
    // for (int i = 0; i < w_out / 2; i++)
    //     printf("%d ", xofs[i]);
    // printf("\n");
    // for (int i = 0; i < h_out; i++)
    //     printf("%d ", yofs[i]);
    // printf("\n");

#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        for (int dy = 0; dy < h_out; dy++){
            unsigned char* out_ptr = out_data + dy * w_out;
            int y_in_start = yofs[dy];
            int y_in_end = y_in_start + 1;
            float b0 = ibeta[dy * 2];
            float b1 = ibeta[dy * 2 + 1];
            for (int dx = 0; dx < w_out; dx += 2){
                int tmp = dx / 2;
                int x_in_start = xofs[tmp] * 2; //0
                int x_in_end = x_in_start + 2; //2
                // printf("x_in: %d, y_in: %d \n", x_in_start, y_in_start);
                float a0 = ialpha[tmp * 2];
                float a1 = ialpha[tmp * 2 + 1];

                int tl_index = y_in_start * w_in + x_in_start; //0
                int tr_index = y_in_start * w_in + x_in_end; //2
                int bl_index = y_in_end * w_in + x_in_start;
                int br_index = y_in_end * w_in + x_in_end;

                int tl = in_data[tl_index + i * spatial_in];
                int tr = in_data[tr_index + i * spatial_in];
                int bl = in_data[bl_index + i * spatial_in];
                int br = in_data[br_index + i * spatial_in];

                float outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;

                out_ptr[dx] = outval;

                tl_index++;
                tr_index++;
                bl_index++;
                br_index++;

                tl = in_data[tl_index + i * spatial_in];
                tr = in_data[tr_index + i * spatial_in];
                bl = in_data[bl_index + i * spatial_in];
                br = in_data[br_index + i * spatial_in];

                outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;

                out_ptr[dx + 1] = outval;

            }
        }
    }
    delete[] ialpha;
    delete[] ibeta;
    delete[] buf;
}

void resize_y_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, float width_scale, float height_scale) {

    // LOG(INFO) << "input w, h:" << w_in << ", " << h_in;
    // LOG(INFO) << "output w, h:" << w_out << ", " << h_out;

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;

    int* buf = new int[w_out * 2 + h_out * 2];
    int* xofs = buf;//new int[w];
    int* yofs = buf + w_out;//new int[h];

    float* ialpha = new float[w_out * 2];//new short[w * 2];
    float* ibeta = new float[h_out * 2];//new short[h * 2];

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;

    for (int dx = 0; dx < w_out; dx++){
        fx = (float)((dx + 0.5) * width_scale - 0.5);
        sx = floor(fx);
        fx -= sx;

        if (sx < 0){
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w_in - 1){
            sx = w_in - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx);
        float a1 = fx;

        ialpha[dx * 2] = a0;
        ialpha[dx * 2 + 1] = a1;
    }

    for (int dy = 0; dy < h_out; dy++) {
        fy = (float)((dy + 0.5) * height_scale - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0){
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1){
            sy = h_in - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy);
        float b1 =        fy;

        ibeta[dy * 2] = b0;
        ibeta[dy * 2 + 1] = b1;
    }

#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        for (int s = 0; s < spatial_out; ++s){
            int x_out = s % w_out;
            int y_out = s / w_out;

            int x_in_start = xofs[x_out]; //(int)x_in;
            int y_in_start = yofs[y_out];

            int x_in_end = x_in_start + 1;
            int y_in_end = y_in_start + 1;

            float a0 = ialpha[x_out * 2];
            float a1 = ialpha[x_out * 2 + 1];
            float b0 = ibeta[y_out * 2];
            float b1 = ibeta[y_out * 2 + 1];

            int tl_index = y_in_start * w_in + x_in_start;
            int tr_index = y_in_start * w_in + x_in_end;
            int bl_index = y_in_end * w_in + x_in_start;
            int br_index = y_in_end * w_in + x_in_end;

            int tl = in_data[tl_index + i * spatial_in];
            int tr = in_data[tr_index + i * spatial_in];
            int bl = in_data[bl_index + i * spatial_in];
            int br = in_data[br_index + i * spatial_in];

            float outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;

            out_data[s + i * spatial_out] = outval;
        }
    }
    delete[] ialpha;
    delete[] ibeta;
   // delete[] buf;

}

void resize_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, float width_scale, float height_scale) {
    if (w_out == w_in && h_out == h_in)
    {
        memcpy(out_data, in_data, sizeof(char) * w_in * w_in);
        return;
    }
   // dst = new unsigned char[h_out * w_out];
    //if (dst == nullptr)
   //     return;
    int y_h = h_in * 2 / 3;
    int uv_h = h_in - y_h;
    const unsigned char* y_ptr = in_data;
    const unsigned char* uv_ptr = in_data + y_h * w_in;
    //out
    int dst_y_h = h_out * 2 / 3;
    int dst_uv_h = h_out - dst_y_h;
    unsigned char* dst_ptr = out_data + dst_y_h * w_out;

    //resize_y_basic(in_data, 1, h_in, w_in, out_data, h_out, w_out, width_scale, height_scale);
    //y
    resize_y_basic(y_ptr, 1, y_h, w_in, out_data, dst_y_h, w_out, width_scale, height_scale);
    //uv
    resize_uv_basic(uv_ptr, 1, uv_h, w_in, dst_ptr, dst_uv_h, w_out, width_scale, height_scale);
}

void nv21_to_tensor_basic(const unsigned char* nv21, TensorHf& output, int width, int height, \
    float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(height, output.height(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(3, output.channel(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(1, output.num(), "sizes of two valid shapes must be the same");
    int size = width * height;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float* ptr1 = static_cast<float*>(output.mutable_data()) + size;
    float* ptr2 = static_cast<float*>(output.mutable_data()) + size * 2;
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];
    const unsigned char* uv_start = nv21 + size;

    for (int h = 0; h < height; h++){
        int y = 0;
        int u = 0;
        int v = 0;
        int size_h = h * width;
        int u_size_h = (h / 2) * width;
        for (int i = 0; i < width; i++){
            y = nv21[size_h + i];
            if (i % 2 == 0){
                v = uv_start[u_size_h + i];
                u = uv_start[u_size_h + i + 1];
            }
            //printf("y0: %d, u: %d, v: %d\n", y, u, v);
            *ptr0 = ((y + 0.14 * (v - 128)) - r_means) * r_scales;
            *ptr1 = ((y - (0.34 * (u - 128)) - (0.71 * (v - 128)))- g_means)  * g_scales;
            *ptr2 = ((y + (1.77 * (u - 128))) - b_means) * b_scales;

            ptr0++;
            ptr1++;
            ptr2++;
        }
    }
}

void nv12_to_tensor_basic(const unsigned char* nv12, TensorHf& output, int width, int height, \
    float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(height, output.height(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(3, output.channel(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(1, output.num(), "sizes of two valid shapes must be the same");
    int size = width * height;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float* ptr1 = static_cast<float*>(output.mutable_data()) + size;
    float* ptr2 = static_cast<float*>(output.mutable_data()) + size * 2;
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];
    const unsigned char* uv_start = nv12 + size;

    float r_meanxscale = r_means * r_scales;
    float g_meanxscale = g_means * g_scales;
    float b_meanxscale = b_means * b_scales;

    for (int h = 0; h < height; h++){
        int y = 0;
        int u = 0;
        int v = 0;
        int size_h = h * width;
        int u_size_h = (h / 2) * width;
        for (int i = 0; i < width; i++){
            y = nv12[size_h + i];
            if (i % 2 == 0){
                u = uv_start[u_size_h + i];
                v = uv_start[u_size_h + i + 1];
            }
            //printf("y0: %d, u: %d, v: %d\n", y, u, v);
            *ptr0 = ((y + 0.14 * (v - 128)) - r_means) * r_scales;
            *ptr1 = ((y - (0.34 * (u - 128)) - (0.71 * (v - 128)))- g_means)  * g_scales;
            *ptr2 = ((y + (1.77 * (u - 128))) - b_means) * b_scales;

            ptr0++;
            ptr1++;
            ptr2++;
        }
    }
}

void rotate90_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[y * w_out + x] = in_data[x * w_in + y]; //(y,x) = in(x,y)
        }
    }
}

void rotate180_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int w = w_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[x * w_out + w - y] = in_data[x * w_in + y]; //(y,x) = in(x,y)
        }
    }
}
void rotate270_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int h = h_out - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[(h - y) * w_out + x] = in_data[x * w_in + y]; //(y,x) = in(x,y)
        }
    }
}

void rotate_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, int angle){
    if (angle == 90){
        LOG(INFO) << "90";
        rotate90_basic(in_data, h_in, w_in, out_data, h_out, w_out);
    }
    if (angle == 180){
        LOG(INFO) << "180";
        rotate180_basic(in_data, h_in, w_in, out_data, h_in, w_in);
    }
    if (angle == 270){
        LOG(INFO) << "270";
        rotate270_basic(in_data, h_in, w_in, out_data, h_out, w_out);
    }
    //LOG(INFO) << "end";

}
void flipx_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int h = h_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[(h - x) * w_in + y] = in_data[x * w_in + y]; //(y,x) = in(x,y)
        }
    }
}

void flipy_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int w = w_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[x * w_in + w - y] = in_data[x * w_in + y]; //(y,x) = in(x,y)
        }
    }
}
void flipxy_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int w = w_in - 1;
    int h = h_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            out_data[(h - x) * w_in + w - y] = in_data[x * w_in + y]; //(h-y,w-x) = in(x,y)
        }
    }
}

void flip_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, int flip_num){
    if (flip_num == 1){ //x
        LOG(INFO) << "x";
        flipx_basic(in_data, h_in, w_in, out_data);
    }
    if (flip_num == -1){
        LOG(INFO) << "y";
        flipy_basic(in_data, h_in, w_in, out_data);
    }
    if (flip_num == 0){
        LOG(INFO) << "xy";
        flipxy_basic(in_data, h_in, w_in, out_data);
    }
    //LOG(INFO) << "end";

}

void nv12_bgr_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int y_h = h_in * 2 / 3;
    const unsigned char* y = in_data;
    const unsigned char* vu = in_data + y_h * w_in;
    for (int i = 0; i < y_h; i++){
        const unsigned char* ptr_y1 = y + i * w_in;
        const unsigned char* ptr_vu = vu + (i / 2) * w_in;
        unsigned char* ptr_bgr1 = out_data + (i * 3) * w_out;
        unsigned char* ptr_bgr2 = ptr_bgr1 + w_out;
        unsigned char* ptr_bgr3 = ptr_bgr2 + w_out;
        int j = 0;
        for (; j < w_in; j += 2){
            unsigned char _y0 = ptr_y1[0];
            unsigned char _y1 = ptr_y1[1];
            unsigned char _v = ptr_vu[1];
            unsigned char _u = ptr_vu[0];

            int ra = floor((179 * (_v - 128)) >> 7);
            int ga = floor((44 * (_u - 128) + 91 * (_v-128)) >> 7);
            int ba = floor((227 * (_u - 128)) >> 7);

            int r = _y0 + ra;
            int g = _y0 - ga;
            int b = _y0 + ba;

            int r1 = _y1 + ra;
            int g1 = _y1 - ga;
            int b1 = _y1 + ba;

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
            g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
            b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

            *ptr_bgr1++ = b;
            *ptr_bgr2++ = g;
            *ptr_bgr3++ = r;

            *ptr_bgr1++ = b1;
            *ptr_bgr2++ = g1;
            *ptr_bgr3++ = r1;

            ptr_y1 += 2;
            ptr_vu += 2;

        }
        if (j < w_in) {
            unsigned char _y = ptr_y1[0];
            unsigned char _v = ptr_vu[1];
            unsigned char _u = ptr_vu[0];

            int r = _y + ((179 * (_v - 128)) >> 7);
            int g = _y - ((44 * (_u - 128) - 91 * (_v-128)) >> 7);
            int b = _y + ((227 * (_u - 128)) >> 7);

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            ptr_bgr1[0] = b;
            ptr_bgr1[1] = g;
            ptr_bgr1[2] = r;
        }
    }
}

void nv21_bgr_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int y_h = h_in * 2 / 3;
    const unsigned char* y = in_data;
    const unsigned char* vu = in_data + y_h * w_in;
    for (int i = 0; i < y_h; i++){
        const unsigned char* ptr_y1 = y + i * w_in;
        const unsigned char* ptr_vu = vu + (i / 2) * w_in;
        unsigned char* ptr_bgr1 = out_data + (i * 3) * w_out;
        unsigned char* ptr_bgr2 = ptr_bgr1 + w_out;
        unsigned char* ptr_bgr3 = ptr_bgr2 + w_out;
        int j = 0;
        for (; j < w_in; j += 2){
            unsigned char _y0 = ptr_y1[0];
            unsigned char _y1 = ptr_y1[1];
            unsigned char _v = ptr_vu[0];
            unsigned char _u = ptr_vu[1];

            int ra = floor((179 * (_v - 128)) >> 7);
            int ga = floor((44 * (_u - 128) + 91 * (_v-128)) >> 7);
            int ba = floor((227 * (_u - 128)) >> 7);

            // float ra_1 = ((179 * (_v - 128)) / 128.0);
            // float ga_1 = ((44 * (_u - 128) + 91 * (_v-128)) / 128.0);
            // float ba_1 = ((227 * (_u - 128)) / 128.0);

            // int ra = ra_1 < 0 ? ceil(ra_1) : floor(ra_1);
            // int ga = ga_1 < 0 ? ceil(ga_1) : floor(ga_1);
            // int ba = ba_1 < 0 ? ceil(ba_1) : floor(ba_1);

            // printf("ga_1, ra, ga, ba: %.3f, %d, %d, %d \n", ga_1, ra, ga, ba);

            int r = _y0 + ra;
            int g = _y0 - ga;
            int b = _y0 + ba;

            int r1 = _y1 + ra;
            int g1 = _y1 - ga;
            int b1 = _y1 + ba;

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
            g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
            b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

            *ptr_bgr1++ = b;
            *ptr_bgr2++ = g;
            *ptr_bgr3++ = r;

            *ptr_bgr1++ = b1;
            *ptr_bgr2++ = g1;
            *ptr_bgr3++ = r1;

            ptr_y1 += 2;
            ptr_vu += 2;

        }
        if (j < w_in) {
            unsigned char _y = ptr_y1[0];
            unsigned char _v = ptr_vu[0];
            unsigned char _u = ptr_vu[1];

            int r = _y + ((179 * (_v - 128)) >> 7);
            int g = _y - ((44 * (_u - 128) - 91 * (_v-128)) >> 7);
            int b = _y + ((227 * (_u - 128)) >> 7);

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            ptr_bgr1[0] = b;
            ptr_bgr1[1] = g;
            ptr_bgr1[2] = r;
        }
    }
}

void bgr_to_tensor_basic(const unsigned char* bgr, TensorHf& output, int width, int height, \
    float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(height, output.height() * 3, "sizes of two valid shapes must be the same");
    LCHECK_EQ(3, output.channel(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(1, output.num(), "sizes of two valid shapes must be the same");
    int size = width * height / 3;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];

    for (int h = 0; h < height; h += 3){
        const unsigned char* ptr_b = bgr + (h * 3) * width;
        const unsigned char* ptr_g = ptr_b + width;
        const unsigned char* ptr_r = ptr_g + width;
        float* ptr0_b = ptr0 + (h / 3)* width;
        float* ptr1_g = ptr0_b + size;
        float* ptr2_r = ptr1_g + size;
        for (int i = 0; i < width; i++){
            *ptr0_b++ = (*ptr_b - b_means) * b_scales;
            *ptr1_g++ = (*ptr_g - g_means) * g_scales;
            *ptr2_r++ = (*ptr_r - r_means) * r_scales;

            *ptr_b++;
            *ptr_g++;
            *ptr_r++;
        }
    }
}
#if 0
TEST(TestSaberLite, test_func_cv_bgr_tensor) {
    LOG(INFO) << "test_func_cv_bgr_tensor start";
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;

    Shape shape_in(1, 1, h_in, w_in);
    Shape shape_out(1, 3, h_in / 3, w_in);

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in ;
    unsigned char* bgr = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        bgr[i] = (unsigned char)i;
    }

    TensorHf4 tout(shape_out);
    TensorHf4 tout_basic(shape_out);

    float means[3] = {127.5f, 127.5f, 127.5f};
    float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};

#if COMPARE_RESULT
   // nv21_to_tensor_basic(nv21, tout_basic, w_in, h_in, means, scales);
    bgr_to_tensor_basic(bgr, tout_basic, w_in, h_in, means, scales);
    //print_tensor(tout_basic);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv bgrtoTensor compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        //nv21_to_tensor(nv21, tout, w_in, h_in, means, scales);
        bgr_to_tensor(bgr, tout, w_in, h_in, means, scales);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber bgrtoTensor total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(tout, tout_basic, max_ratio, max_diff);

    TensorHf4 diff(shape_out);
    tensor_diff(tout_basic, tout, diff);
    if (fabsf(max_ratio) > 1e-3f) {
        LOG(INFO) << "diff: ";
        print_tensor(diff);
    }
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}
#endif
#if 0
TEST(TestSaberLite, test_func_cv_nv21_bgr) {
    LOG(INFO) << "test_func_cv_nv21_bgr start";
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;
    // int w_out = ww;
    // int h_out = hh;
    int w_out = w_in;
    int h_out = h_in * 2;

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << " flip_num = " << flip_num;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in;
    unsigned char* nv21 = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        nv21[i] = (unsigned char)(i + 10);
    }
    unsigned char* out = new unsigned char[size * 3];
    unsigned char* tv_out = new unsigned char[size * 3];

#if COMPARE_RESULT
    //nv21_bgr_basic(nv21, 1, h_in, w_in, out, h_out, w_out);
    nv12_bgr_basic(nv21, 1, h_in, w_in, out, h_out, w_out);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv flip compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        //nv21_to_bgr(nv21, tv_out, w_in, h_in, w_out, h_out);
        nv12_to_bgr(nv21, tv_out, w_in, h_in, w_out, h_out);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber flip total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    const double eps = 1e-6f;
    LOG(INFO) << "diff: " ;
    size = w_out * h_out;
    for (int i = 0; i < size; i++){
        int a = out[i];
        int b = tv_out[i];
        int diff1 = a - b;
        int diff = diff1 >= 0 ? diff1 : -1 * diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % w_out == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    printf("\n");
    if (fabsf(max_ratio) > 1e-5f){
        LOG(INFO) << "in";
        for (int i = 0; i < h_in; i++){
            for (int j = 0; j < w_in; j++){
                printf("%d  ", nv21[i*w_in+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", out[i*w_out+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "tv_out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", tv_out[i*w_out+j]);
            }
            printf("\n");
        }

    }

    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
    delete[] out;
    delete[] tv_out;
}
#endif
#if 0
TEST(TestSaberLite, test_func_cv_flip) {
    LOG(INFO) << "test_func_cv_flip start";
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;
    int w_out = ww;
    int h_out = hh;

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) <<" flip_num = "<< flip_num;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in;
    unsigned char* nv21 = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        nv21[i] = (unsigned char)i;
    }
    unsigned char* out = new unsigned char[size];
    unsigned char* tv_out = new unsigned char[size];


#if COMPARE_RESULT
   // nv21_to_tensor_basic(nv21, tout_basic, w_in, h_in, means, scales);
    flip_basic(nv21, 1, h_in, w_in, out, h_out, w_out, flip_num);
    //print_tensor(tout_basic);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv flip compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        //nv21_to_tensor(nv21, tout, w_in, h_in, means, scales);
        flip(nv21, tv_out, w_in, h_in, w_out, h_out, flip_num);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber flip total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    const double eps = 1e-6f;
    LOG(INFO) << "diff: " ;
    for (int i = 0; i < size; i++){
        int a = out[i];
        int b = tv_out[i];
        int diff1 = a - b;
        int diff = diff1 >= 0 ? diff1 : -1 * diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        if (i != 0 && i % w_out == 0)
            printf("\n");
        printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    printf("\n");
    if (fabsf(max_ratio) > 1e-5f){
        LOG(INFO) << "in";
        for (int i = 0; i < h_in; i++){
            for (int j = 0; j < w_in; j++){
                printf("%d  ", nv21[i*w_in+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", out[i*w_out+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "tv_out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", tv_out[i*w_out+j]);
            }
            printf("\n");
        }
    }

    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
    delete[] out;
    delete[] tv_out;
}
#endif
#if 0
TEST(TestSaberLite, test_func_cv_rotate) {
    LOG(INFO) << "test_func_cv_rotate start";
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;
    int w_out = ww;
    int h_out = hh;

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) <<" angle = "<< angle;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in;
    unsigned char* nv21 = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        nv21[i] = (unsigned char)i;
    }
    unsigned char* out = new unsigned char[size];
    unsigned char* tv_out = new unsigned char[size];


#if COMPARE_RESULT
   // nv21_to_tensor_basic(nv21, tout_basic, w_in, h_in, means, scales);
    rotate_basic(nv21, 1, h_in, w_in, out, h_out, w_out, angle);
    //print_tensor(tout_basic);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv rotate compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        //nv21_to_tensor(nv21, tout, w_in, h_in, means, scales);
        rotate(nv21, tv_out, w_in, h_in, w_out, h_out, angle);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber rotate total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    const double eps = 1e-6f;
    LOG(INFO) << "diff: " ;
    for (int i = 0; i < size; i++){
        int a = out[i];
        int b = tv_out[i];
        int diff1 = a - b;
        int diff = diff1 >= 0 ? diff1 : -1 * diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        if (i != 0 && i % w_out == 0)
            printf("\n");
        printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    printf("\n");
    if (fabsf(max_ratio) > 1e-5f){
        LOG(INFO) << "in";
        for (int i = 0; i < h_in; i++){
            for (int j = 0; j < w_in; j++){
                printf("%d  ", nv21[i*w_in+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", out[i*w_out+j]);
            }
            printf("\n");
        }
        LOG(INFO) << "tv_out";
        for (int i = 0; i < h_out; i++){
            for (int j = 0; j < w_out; j++){
                printf("%d  ", tv_out[i*w_out+j]);
            }
            printf("\n");
        }
    }

    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
    delete[] out;
    delete[] tv_out;
}
#endif
#if 0
TEST(TestSaberLite, test_func_cv_resize) {
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;
    int w_out = ww;
    int h_out = hh;

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << " output tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_out << ", width=" << w_out;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in;
    unsigned char* nv21 = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        nv21[i] = (unsigned char)i;
    }

    int out_size = h_out * w_out;
    unsigned char* tout = new unsigned char[out_size];
    unsigned char* tout_basic = new unsigned char[out_size];

    float width_scale = (float)w_in / w_out;
    float height_scale = (float)h_in / h_out;

#if COMPARE_RESULT
    LOG(INFO) << "saber cv basic resize compute";
    resize_basic(nv21, 1, h_in, w_in, tout_basic, h_out, w_out, width_scale, height_scale);
    //print_tensor(tout_basic);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv resize compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
       // LOG(INFO) << "resize";
        resize(nv21, tout, w_in, h_in, w_out, h_out);

        LOG(INFO) << "nv21";
        Shape shape_out = {1, 3, w_out, h_out * 2/3};
        TensorHf4 tout_tensor(shape_out);
        float means[3] = {127.5f, 127.5f, 127.5f};
        float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
        nv12_to_tensor(tout, tout_tensor, w_out, h_out * 2/3, means, scales);

        LOG(INFO) << "end";
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber resize total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    // LOG(INFO) << "basic result, size: " << out_size;
    // for (int i = 0; i < out_size; i++){
    //     if (i != 0 && i % w_out == 0)
    //         printf("\n");
    //     printf("%d   ", tout_basic[i]);
    // }
    // printf("\n");
    // LOG(INFO) << "resize result, size: " << out_size;
    // for (int i = 0; i < out_size; i++){
    //     if (i != 0 && i % w_out == 0)
    //         printf("\n");
    //     printf("%d   ", tout[i]);
    // }
    // printf("\n");
    //tensor_cmp_host(tout_basic, tout, out_size, max_ratio, max_diff);
    const double eps = 1e-6f;
    LOG(INFO) << "diff, size: " << out_size;
    for (int i = 0; i < out_size; i++){
        int a = tout[i];
        int b = tout_basic[i];
        int diff1 = a - b;
        int diff = diff1 >= 0 ? diff1 : -1 * diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % w_out == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    printf("\n");
    // LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    // CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
    delete[] tout;
    delete[] tout_basic;
   // LOG(INFO) << "resize end";
}
#endif

#if 0
TEST(TestSaberLite, test_func_cv_nv21_tensor) {
    LOG(INFO) << "test_func_cv_nv21_tensor start";
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int test_iter = 1;

    int w_in = w;
    int h_in = h;

    Shape shape_in(1, 1, h_in, w_in);
    Shape shape_out(1, 3, h_in, w_in);

    LOG(INFO) << " input tensor size, num=" << 1 << ", channel=" << \
        1 << ", height=" << h_in << ", width=" << w_in;

    //Tensor<CPU, AK_UINT8> thin(shape_in);
    int size = h_in * w_in * 3;
    size = size >> 1;
    unsigned char* nv21 = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        nv21[i] = (unsigned char)i;
    }

    TensorHf4 tout(shape_out);
    TensorHf4 tout_basic(shape_out);

    float means[3] = {127.5f, 127.5f, 127.5f};
    float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};

#if COMPARE_RESULT
   // nv21_to_tensor_basic(nv21, tout_basic, w_in, h_in, means, scales);
    nv12_to_tensor_basic(nv21, tout_basic, w_in, h_in, means, scales);
    //print_tensor(tout_basic);
#endif

    SaberTimer t1;

    LOG(INFO) << "saber cv nv21toTensor compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        //nv21_to_tensor(nv21, tout, w_in, h_in, means, scales);
        nv12_to_tensor(nv21, tout, w_in, h_in, means, scales);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber nv21toTensor total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(tout);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(tout_basic.data(), tout.data(), tout_basic.valid_size(), max_ratio, max_diff);
    TensorHf4 diff(shape_out);
    tensor_diff(tout_basic, tout, diff);
    if (fabsf(max_ratio) > 1e-3f) {
        LOG(INFO) << "diff: ";
        print_tensor(diff);
    }
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}
#endif
int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
   // Env::env_init(4);
    Env::env_init();
    LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  h " << \
                " w hh ww angle";
    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }

    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    if (argc >= 4) {
        h = atoi(argv[3]);
    }
    if (argc >= 5) {
        w = atoi(argv[4]);
    }
    if (argc >= 6) {
        hh = atoi(argv[5]);
    }
    if (argc >= 7) {
        ww = atoi(argv[6]);
    }
    if (argc >= 8){
        flip_num = atoi(argv[7]);
    }
    if (argc >= 9){
        angle = atoi(argv[8]);
    }

    InitTest();
    //RUN_ALL_TESTS(argv[0]);
    return 0;
}

