#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "test_saber_base.h"
#include "saber/saber_types.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/debug.h"
#include "saber/funcs/detection_output.h"
#include <vector>
#include <string>
using namespace anakin::saber;
#if defined(USE_CUDA)
using Target = NV;
using Target_H = NVHX86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#endif

std::string g_bbox_file = "/home/public/multiclass_nms/result_box_clip_0.tmp_0.txt";
std::string g_conf_file = "/home/public/multiclass_nms/result_softmax_0.tmp_0.txt";
std::string g_priorbox_file = "";
std::string g_result_file = "/home/public/multiclass_nms/result_multiclass_nms_0.tmp_0.txt";
std::string g_img_file = "/home/public/000000000139.jpg";

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"

using namespace cv;

struct Object{
    int batch_id;
    cv::Rect rec;
    int class_id;
    float prob;
};

void detect_object(Tensor<Target_H>& tout, const float thresh, std::vector<cv::Mat>& image, const std::string& name) {
    int img_num = image.size();
    const float* dout = static_cast<const float*>(tout.data());
    std::vector<Object> objects;
    for (int iw = 0; iw < tout.height(); iw++) {
        Object object;
        const float *values = dout + iw * tout.width();
        int batch_id = static_cast<int>(values[0]);
        object.batch_id = batch_id;
        object.class_id = (int)values[1];
        object.prob = values[2];
        object.rec.x = (int)(values[3]);
        object.rec.y = (int)(values[4]);
        object.rec.width = (int)(values[5] - values[3]);
        object.rec.height = (int)(values[6] - values[4]);
        objects.push_back(object);
    }

    for (int i = 0; i < objects.size(); ++i) {
        Object object = objects.at(i);
        if (object.prob > thresh && object.batch_id < image.size()) {
            cv::rectangle(image[object.batch_id], object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str << "class: " << object.class_id << " + score: " << object.prob;
            cv::putText(image[object.batch_id], pro_str.str(), cv::Point(object.rec.x, object.rec.y), \
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            LOG(INFO) << "detection in batch: " << object.batch_id << ", image size: " << \
                    image[object.batch_id].cols << ", " << image[object.batch_id].rows << \
                    ", detect object: " << object.class_id << ", location: x=" << \
                    object.rec.x << ", y=" << object.rec.y << ", width=" << object.rec.width << \
                    ", height=" << object.rec.height;
        }
    }
    for (int j = 0; j < image.size(); ++j) {
        std::ostringstream str;
        str << name << "_detection_out_" << j << ".jpg";
        cv::imwrite(str.str(), image[j]);
    }
}
#endif

template <typename dtype>
static bool sort_score_pair_descend(const std::pair<float, dtype>& pair1, \
                                    const std::pair<float, dtype>& pair2) {
    return pair1.first > pair2.first;
}

template <typename dtype>
void get_max_score_index(const dtype* scores, int num, std::vector<std::pair<dtype, int> >* score_index_vec) {
    //! Generate index score pairs.
    for (int i = 0; i < num; ++i) {
        score_index_vec->push_back(std::make_pair(scores[i], i));
    }

    //! Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(), \
                     sort_score_pair_descend<int>);
}

void sort_result(const float* res, int count, Tensor<Target_H>& tout, const std::vector<int>& offset = {}) {
    std::vector<std::pair<int, std::vector<float>>> vres;
    tout.reshape(Shape({1, 1, count / 6, 7}, Layout_NCHW));
    float* dout = static_cast<float*>(tout.mutable_data());
    int batch_size = 1;
    if (offset.size() > 0) {
        batch_size = offset.size() - 1;
    }
    for (int k = 0; k < batch_size; ++k) {
        int batch_id = k;
        int cls_id = -1;
        std::vector<float> score;
        for (int i = 0; i < count; i += 6) {
            int id = static_cast<int>(res[i]);
            if (cls_id >= 0) {
                if (id != cls_id) {
                    vres.emplace_back(std::make_pair(cls_id, score));
                    cls_id = id;
                    score.clear();
                    score.push_back(res[i + 1]);
                } else {
                    score.push_back(res[i + 1]);
                }
            } else {
                cls_id = id;
                score.clear();
                score.push_back(res[i + 1]);
            }
        }
        vres.emplace_back(std::make_pair(cls_id, score));
        LOG(INFO) << "num of classes: " << vres.size();
        const float* din = res;
        for (int j = 0; j < vres.size(); ++j) {
            float* scores = vres[j].second.data();
            int count = vres[j].second.size();
            std::vector<std::pair<float, int>> score_index_vec;
            get_max_score_index(scores, count, &score_index_vec);
            for (int i = 0; i < score_index_vec.size(); ++i) {
                *(dout++) = batch_id;
                *(dout++) = vres[j].first;
                *(dout++) = score_index_vec[i].first;
                *(dout++) = din[score_index_vec[i].second * 6 + 2];
                *(dout++) = din[score_index_vec[i].second * 6 + 3];
                *(dout++) = din[score_index_vec[i].second * 6 + 4];
                *(dout++) = din[score_index_vec[i].second * 6 + 5];
            }
            din += score_index_vec.size() * 6;
        }
    }
}

TEST(TestSaberFunc, test_func_detection_output) {
    const int batch0_start = 0;
    const int batch0_end = 112;
    std::vector<int> offset = {batch0_start, batch0_end};
    std::vector<std::vector<int>> seq_offset;
    seq_offset.push_back(offset);
    Shape shbbox({batch0_end - batch0_start, 81, 4, 1}, Layout_NCHW);
    Shape shconf({batch0_end - batch0_start, 81, 1, 1}, Layout_NCHW);
    Shape shres({1, 1, 112, 7}, Layout_NCHW);
    Tensor<Target_H> thbbox(shbbox);
    Tensor<Target_H> thconf(shconf);
    Tensor<Target_H> thres_gt(shres);
    Tensor<Target> tdbbox(shbbox);
    Tensor<Target> tdconf(shconf);
    Tensor<Target> tdres(shres);
    Tensor<Target_H> thres(shres);

    std::vector<float> vbbox;
    std::vector<float> vconf;
    std::vector<float> vres;
    if (!read_file(vbbox, g_bbox_file.c_str())) {
        LOG(ERROR) << "load bbox file failed";
        return;
    }
    if (!read_file(vconf, g_conf_file.c_str())) {
        LOG(ERROR) << "load conf file failed";
        return;
    }
    if (!read_file(vres, g_result_file.c_str())) {
        LOG(ERROR) << "load ground truth failed";
        return;
    }

    thres_gt.reshape(Shape({1, 1, vres.size() / 6, 6}, Layout_NCHW));

    memcpy(thbbox.mutable_data(), vbbox.data(), sizeof(float) * vbbox.size());
    memcpy(thconf.mutable_data(), vconf.data(), sizeof(float) * vconf.size());
    memcpy(thres_gt.mutable_data(), vres.data(), sizeof(float) * vres.size());

    //! sort the ground truth
    sort_result(static_cast<const float *>(thres_gt.data()), thres_gt.valid_size(), thres);
    print_tensor_valid(thres);
//    print_tensor_valid(thbbox);
//    print_tensor_valid(thconf);
//    print_tensor_valid(thres);
    tdbbox.copy_from(thbbox);
    tdconf.copy_from(thconf);
    tdbbox.set_seq_offset(seq_offset);
    tdconf.set_seq_offset(seq_offset);

    //! init params
    DetectionOutputParam<Target> det_param;
    det_param.background_id = 0;
    det_param.share_location = false;
    det_param.class_num = 0;
    det_param.type = CORNER;
    det_param.conf_thresh = 0.05f;
    det_param.keep_top_k = 100;
    det_param.variance_encode_in_target = false;
    det_param.nms_eta = 1.f;
    det_param.nms_top_k = -1;
    det_param.nms_thresh = 0.5f;

    //! create op
    DetectionOutput<Target, AK_FLOAT> det_op;

    //! create io
    std::vector<Tensor<Target> *> input_v;
    std::vector<Tensor<Target> *> output_v;
    input_v.push_back(&tdbbox);
    input_v.push_back(&tdconf);
    output_v.push_back(&tdres);

    //! create context
    Context<Target> ctx;

    //! init op
    det_op.compute_output_shape(input_v, output_v, det_param);
    output_v[0]->reshape(output_v[0]->valid_shape());
    SABER_CHECK(det_op.init(input_v, output_v, det_param, SPECIFY, SABER_IMPL, ctx));

    //! op dispatch
    SABER_CHECK(det_op(input_v, output_v, det_param, ctx));
    print_tensor_valid(*output_v[0]);

    Tensor<Target_H> thres_res(output_v[0]->valid_shape());
    thres_res.copy_from(*output_v[0]);

#ifdef USE_OPENCV
    cv::Mat img = cv::imread(g_img_file);
    if (img.empty()) {
        return;
    }
    cv::Mat img_gt = img.clone();
    cv::Mat img_res = img.clone();
    std::vector<cv::Mat> v_gt = {img_gt};
    std::vector<cv::Mat> v_res = {img_res};
    LOG(INFO) << "draw gt box to image";
    detect_object(thres, 0.05f, v_gt, "gt");
    LOG(INFO) << "draw test box to image";
    detect_object(thres_res, 0.05f, v_res, "test");
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    Env<Target>::env_init();
    Env<Target_H>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

