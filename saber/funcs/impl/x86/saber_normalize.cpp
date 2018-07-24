template <typename dtype>
void norm_cpu_nchw(const int p, const dtype* scale, const dtype* src, dtype* dst, \
                   bool across_spatial, bool has_scale, bool channel_shared, float eps, \
                   int n, int c, int h, int w) {

    const dtype* src_ptr = src;
    dtype* dst_ptr = dst;

    if (across_spatial) {
        int compute_size = h * w * c;
        int outer_size = n * c * h * w / compute_size;

        for (int i = 0; i < outer_size; ++i) {
            dtype sum = 0;

            for (int j = 0; j < compute_size; ++j) {
                if (p == 1) {
                    sum += fabsf(src_ptr[j]);
                } else {
                    sum += src_ptr[j] * src_ptr[j];
                }
            }

            //LOG(INFO) << "idx: " << i << ", " << "norm: " << sum;

            if (p == 1) {
                sum = 1 / (sum + eps);
            } else {
                sum = 1 / (sqrtf(sum) + eps);
            }

            if (has_scale) { //! with scale
                if (channel_shared) { // scale is shared across channel
                    for (int j = 0; j < compute_size; ++j) {
                        dst_ptr[j] = src_ptr[j] * sum * scale[0];
                    }
                } else {
                    for (int j = 0; j < compute_size; ++j) {
                        int c_idx = j / (h * w);
                        dst_ptr[j] = src_ptr[j] * sum * scale[c_idx];
                    }
                }
            } else { //! without scale
                for (int j = 0; j < compute_size; ++j) {
                    dst_ptr[j] = src_ptr[j] * sum;
                }
            }

            src_ptr += compute_size;
            dst_ptr += compute_size;
        }
    } else {
        int channel_in_size = h * w;

        for (int i = 0; i < n; ++i) {
            const dtype* src_batch_ptr = src_ptr + i * c * h * w;
            dtype* dst_batch_ptr = dst_ptr + i * c * h * w;

            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < w; ++k) {
                    const dtype* src_pixel = src_batch_ptr + 0 * channel_in_size + j * w + k;
                    dtype* dst_pixel = dst_batch_ptr + 0 * channel_in_size + j * w + k;
                    float norm = 0.f;

                    for (int l = 0; l < c; ++l) {
                        if (p == 1) {
                            norm += fabsf(src_pixel[l * channel_in_size]);
                        } else {
                            norm += src_pixel[l * channel_in_size] * src_pixel[l * channel_in_size];
                        }
                    }

                    if (p == 1) {
                        norm = 1.f / (norm + eps);
                    } else {
                        norm = 1.f / (sqrtf(norm) + eps);
                    }

                    for (int l = 0; l < c; ++l) {
                        if (has_scale) {
                            if (channel_shared) {
                                dst_pixel[l * channel_in_size] = \
                                                                 src_pixel[l * channel_in_size] * norm * scale[0];
                            } else {
                                dst_pixel[l * channel_in_size] = \
                                                                 src_pixel[l * channel_in_size] * norm * scale[l];
                            }
                        } else {
                            dst_pixel[l * channel_in_size] = \
                                                             src_pixel[l * channel_in_size] * norm;
                        }
                    }
                }
            }
        }
    }
}