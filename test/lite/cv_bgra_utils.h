#include "test_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

typedef Tensor<CPU> TensorHf;

void nv12_bgra_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int y_h = h_in;
    const unsigned char* y = in_data;
    const unsigned char* vu = in_data + y_h * w_in;
    int wout = w_out * 4;
    for (int i = 0; i < y_h; i++){
        const unsigned char* ptr_y1 = y + i * w_in;
        const unsigned char* ptr_vu = vu + (i / 2) * w_in;
        unsigned char* ptr_bgr1 = out_data + i * wout;
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
            *ptr_bgr1++ = g;
            *ptr_bgr1++ = r;
            *ptr_bgr1++ = 255;

            *ptr_bgr1++ = b1;
            *ptr_bgr1++ = g1;
            *ptr_bgr1++ = r1;
            *ptr_bgr1++ = 255;

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
            ptr_bgr1[3] = 255;
        }
    }
}

void nv21_bgra_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int y_h = h_in;
    const unsigned char* y = in_data;
    const unsigned char* vu = in_data + y_h * w_in;
    int wout = w_out * 4;
    for (int i = 0; i < y_h; i++){
        const unsigned char* ptr_y1 = y + i * w_in;
        const unsigned char* ptr_vu = vu + (i / 2) * w_in;
        unsigned char* ptr_bgr1 = out_data + i * wout;
        int j = 0;
        for (; j < w_in; j += 2){
            unsigned char _y0 = ptr_y1[0];
            unsigned char _y1 = ptr_y1[1];
            unsigned char _v = ptr_vu[0];
            unsigned char _u = ptr_vu[1];

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
            *ptr_bgr1++ = g;
            *ptr_bgr1++ = r;
            *ptr_bgr1++ = 255;

            *ptr_bgr1++ = b1;
            *ptr_bgr1++ = g1;
            *ptr_bgr1++ = r1;
            *ptr_bgr1++ = 255;

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
            ptr_bgr1[3] = 255;
        }
    }
}

void bgra_resize_basic(const unsigned char* in_data, int count, int h_in, int win, \
            unsigned char* out_data, int h_out, int wout, float width_scale, float height_scale) {
    int w_in = win * 4;
    int w_out = wout * 4;
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
    for (int dx = 0; dx < w_out / 4; dx++){
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
        xofs[dx] = sx * 4;
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
        for (int dy = 0; dy < h_out; dy++){
            unsigned char* out_ptr = out_data + dy * w_out;
            int y_in_start = yofs[dy];
            int y_in_end = y_in_start + 1;
            float b0 = ibeta[dy * 2];
            float b1 = ibeta[dy * 2 + 1];
            for (int dx = 0; dx < w_out; dx += 4){
                int tmp = dx / 4;
                int x_in_start = xofs[tmp]; //0
                int x_in_end = x_in_start + 4; //2
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
                out_ptr[dx] = ceil(outval);
                tl_index++;
                tr_index++;
                bl_index++;
                br_index++;
                tl = in_data[tl_index + i * spatial_in];
                tr = in_data[tr_index + i * spatial_in];
                bl = in_data[bl_index + i * spatial_in];
                br = in_data[br_index + i * spatial_in];
                outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;
                out_ptr[dx + 1] = ceil(outval);
                tl_index++;
                tr_index++;
                bl_index++;
                br_index++;
                tl = in_data[tl_index + i * spatial_in];
                tr = in_data[tr_index + i * spatial_in];
                bl = in_data[bl_index + i * spatial_in];
                br = in_data[br_index + i * spatial_in];
                outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;
                out_ptr[dx + 2] = ceil(outval);
                tl_index++;
                tr_index++;
                bl_index++;
                br_index++;
                tl = in_data[tl_index + i * spatial_in];
                tr = in_data[tr_index + i * spatial_in];
                bl = in_data[bl_index + i * spatial_in];
                br = in_data[br_index + i * spatial_in];
                outval = (tl * a0 + tr * a1) * b0  + (bl * a0 + br * a1) * b1;
                out_ptr[dx + 3] = ceil(outval);
            }
        }
    }
    delete[] ialpha;
    delete[] ibeta;
    delete[] buf;
}

void bgra_to_tensor_hwc_basic(const unsigned char* bgr, TensorHf& output, int width, int height, \
    float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(height, output.height(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(3, output.channel(), "sizes of two valid shapes must be the same");
    LCHECK_EQ(1, output.num(), "sizes of two valid shapes must be the same");
    int size = width * height;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];

    for (int h = 0; h < height; h++){
        const unsigned char* ptr_bgr = bgr + h * width * 4;
        float* ptr_b = ptr0 + h * width;
        float* ptr_g = ptr_b + size;
        float* ptr_r = ptr_g + size;
        for (int i = 0; i < width; i++){
            *ptr_b++ = (*ptr_bgr - b_means) * b_scales;
            *ptr_bgr++;
            *ptr_g++ = (*ptr_bgr - g_means) * g_scales;
            *ptr_bgr++;
            *ptr_r++ = (*ptr_bgr - r_means) * r_scales;
            *ptr_bgr++;
            *ptr_bgr++;
        }
    }
}

void rotate90_hwc_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int win = w_in * 4;
    int wout = w_out * 4;
    // unsigned char* out_data = new unsigned char[h_out * w_out * 3];
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            int tmpx = (w_out - 1 - x) * 4; //x * 3; //
            out_data[y * wout + tmpx] = in_data[x * win + tmpy]; //(w-y,x) = in(x,y)
            out_data[y * wout + tmpx + 1] = in_data[x * win + tmpy + 1]; //(y,x) = in(x,y)
            out_data[y * wout + tmpx + 2] = in_data[x * win + tmpy + 2]; //(y,x) = in(x,y)
            out_data[y * wout + tmpx + 3] = in_data[x * win + tmpy + 3]; //(y,x) = in(x,y)
        }
    }
    // bgr_flip_hwc_basic(out_data, 1, h_out, w_out, out_ptr, h_out, w_out, -1);
    // delete[] out_data;
}

void rotate180_hwc_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int w = w_in * 4 - 1;
    int h = h_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            out_data[(h - x) * w_in * 4 + w - tmpy - 3] = in_data[x * w_in * 4 + tmpy]; //(y,x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy - 2] = in_data[x * w_in * 4 + tmpy + 1]; //(y,x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy - 1] = in_data[x * w_in * 4 + tmpy + 2]; //(y,x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy] = in_data[x * w_in * 4 + tmpy + 3]; //(y,x) = in(x,y)
        }
    }
}
void rotate270_hwc_basic(const unsigned char* in_data, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out){
    int win = w_in * 4;
    int wout = w_out * 4;
    int h = h_out - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            int tmpx = x * 4;
            out_data[(h - y) * wout + tmpx] = in_data[x * win + tmpy]; //(y,x) = in(x,y)
            out_data[(h - y) * wout + tmpx + 1] = in_data[x * win + tmpy + 1]; //(y,x) = in(x,y)
            out_data[(h - y) * wout + tmpx + 2] = in_data[x * win + tmpy + 2]; //(y,x) = in(x,y)
            out_data[(h - y) * wout + tmpx + 3] = in_data[x * win + tmpy + 3]; //(y,x) = in(x,y)
        }
    }
}

void bgra_rotate_hwc_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, int angle){
    if (angle == 90){
        LOG(INFO) << "90";
        rotate90_hwc_basic(in_data, h_in, w_in, out_data, h_out, w_out);
    }
    if (angle == 180){
        LOG(INFO) << "180";
        rotate180_hwc_basic(in_data, h_in, w_in, out_data, h_in, w_in);
    }
    if (angle == 270){
        LOG(INFO) << "270";
        rotate270_hwc_basic(in_data, h_in, w_in, out_data, h_out, w_out);
    }
    //LOG(INFO) << "end";
}
void flipx_hwc_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int h = h_in - 1;
    int w = w_in * 4;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            out_data[(h - x) * w + tmpy] = in_data[x * w + tmpy]; //(y,x) = in(x,y)
            out_data[(h - x) * w + tmpy + 1] = in_data[x * w + tmpy + 1]; //(y,x) = in(x,y)
            out_data[(h - x) * w + tmpy + 2] = in_data[x * w + tmpy + 2]; //(y,x) = in(x,y)
            out_data[(h - x) * w + tmpy + 3] = in_data[x * w + tmpy + 3]; //(y,x) = in(x,y)
        }
    }
}

void flipy_hwc_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int w = w_in * 4 - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            out_data[x * w_in * 4 + w - tmpy - 3] = in_data[x * w_in * 4 + tmpy]; //(y,x) = in(x,y)
            out_data[x * w_in * 4 + w - tmpy - 2] = in_data[x * w_in * 4 + tmpy + 1]; //(y,x) = in(x,y)
            out_data[x * w_in * 4 + w - tmpy - 1] = in_data[x * w_in * 4 + tmpy + 2]; //(y,x) = in(x,y)
            out_data[x * w_in * 4 + w - tmpy] = in_data[x * w_in * 4 + tmpy + 3]; //(y,x) = in(x,y)
        }
    }
}
void flipxy_hwc_basic(const unsigned char* in_data, int h_in, int w_in, unsigned char* out_data){
    int w = w_in * 4 - 1;
    int h = h_in - 1;
    for (int x = 0; x < h_in; x++){
        for (int y = 0; y < w_in; y++){
            int tmpy = y * 4;
            out_data[(h - x) * w_in * 4 + w - tmpy - 3] = in_data[x * w_in * 4 + tmpy]; //(h-y,w-x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy - 2] = in_data[x * w_in * 4 + tmpy + 1]; //(h-y,w-x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy - 1] = in_data[x * w_in * 4 + tmpy + 2]; //(h-y,w-x) = in(x,y)
            out_data[(h - x) * w_in * 4 + w - tmpy] = in_data[x * w_in * 4 + tmpy + 3]; //(h-y,w-x) = in(x,y)
        }
    }
}

void bgra_flip_hwc_basic(const unsigned char* in_data, int count, int h_in, int w_in, \
            unsigned char* out_data, int h_out, int w_out, int flip_num){
    if (flip_num == 1){ //x
        LOG(INFO) << "x";
        flipx_hwc_basic(in_data, h_in, w_in, out_data);
    }
    if (flip_num == -1){
        LOG(INFO) << "y";
        flipy_hwc_basic(in_data, h_in, w_in, out_data);
    }
    if (flip_num == 0){
        LOG(INFO) << "xy";
        flipxy_hwc_basic(in_data, h_in, w_in, out_data);
    }
    //LOG(INFO) << "end";

}
