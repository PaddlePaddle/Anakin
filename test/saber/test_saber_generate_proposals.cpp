#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/generate_proposals.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <fstream>
#include<cmath>
//#define TEST_GENERATE_PROPOSALS
#ifdef TEST_GENERATE_PROPOSALS

using namespace anakin::saber;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

void read_tensor_from_file(float* data, int length, const char* path) {
    std::fstream fs(path);
    int i = 0;
    if (fs.is_open()) {
        std::string str;
        while (true) {
            std::getline(fs, str);
            std::size_t found = str.find(" ");
            if (found != std::string::npos) {
                std::cout << "first 'needle' found at: " << found << '\n';
                break;
            }
            data[i++] = (atof)(str.c_str());
        }
        fs.close();
    }
}

/*NCHW->NHWC*/
template <typename Dtype, typename TargetType_H>
static inline void trans(Tensor<TargetType_H>* out, Tensor<TargetType_H>* in) {
    auto shape = in->valid_shape();
    out->reshape(Shape({shape[0], shape[2], shape[3], shape[1]}, Layout_NCHW));
    auto stride = in->get_stride();
    auto dst = (Dtype*) out->mutable_data();
    auto src = (const Dtype*) in->data();
    for (auto i = 0; i < shape.count(); i++) {
        int n = i / stride[0];
        int c = (i / stride[1]) % shape[1];
        int hw = i % (stride[1]);
        int out_id = n * stride[0] + hw*shape[1] + c;
        dst[out_id] = src[i];
    }
}

template <typename Dtype, typename TargetType_H>
static inline void box_coder(Tensor<TargetType_H>* proposals,
                             const Tensor<TargetType_H>* anchors,
                             const Tensor<TargetType_H>* bbox_deltas,
                             const Tensor<TargetType_H>* variances,
                             std::vector<int>& index
                             ) {
    proposals->reshape(Shape({index.size(), 4, 1, 1}, Layout_NCHW));
    int anchor_nums = index.size();
    int len = anchors->shape()[3];
    CHECK_EQ(len, 4) << "anchor length is 4";
    auto anchor_data = (const Dtype*) anchors->data();
    auto bbox_deltas_data = (const Dtype*) bbox_deltas->data();
    auto proposals_data = (Dtype*) proposals->data();
    const Dtype *variances_data = nullptr;
    if (variances) {
        variances_data = (const Dtype*)variances->data();
    }
    for (int i = 0; i < index.size(); i++) {
        int offset = index[i] * len;
        auto anchor_data_tmp = anchor_data + offset;
        auto variances_data_tmp = variances_data + offset;
        auto bbox_deltas_data_tmp = bbox_deltas_data + offset;
        auto proposals_data_tmp = proposals_data + i*len;
        auto anchor_width = anchor_data_tmp[2] - anchor_data_tmp[0] + 1.0;
        auto anchor_height = anchor_data_tmp[3] - anchor_data_tmp[1] + 1.0;
        auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
        auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;
        Dtype bbox_center_x = 0, bbox_center_y = 0;
        Dtype bbox_width = 0, bbox_height = 0;
        if (variances) {
            bbox_center_x =
                variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
                anchor_center_x;
            bbox_center_y = variances_data_tmp[1] *
                   bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
            bbox_width = std::exp(std::min<Dtype>(variances_data_tmp[ 2] *
                   bbox_deltas_data_tmp[2],
                   kBBoxClipDefault)) * anchor_width;
            bbox_height = std::exp(std::min<Dtype>(variances_data_tmp[3] *
                   bbox_deltas_data_tmp[3],
                   kBBoxClipDefault)) * anchor_height;
        } else {
            bbox_center_x =
                bbox_deltas_data_tmp[0] * anchor_width + anchor_center_x;
            bbox_center_y =
                bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
            bbox_width = std::exp(std::min<Dtype>(bbox_deltas_data_tmp[2],
                    kBBoxClipDefault)) * anchor_width;
            bbox_height = std::exp(std::min<Dtype>(bbox_deltas_data_tmp[3],
                    kBBoxClipDefault)) * anchor_height;
        }
        proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
        proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
        proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - 1;
        proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - 1;
    }
}

template <typename Dtype, typename TargetType_H>
static inline void clip_tiled_boxes(Tensor<TargetType_H> *boxes,
                                   const Tensor<TargetType_H> *im_info) {
  Dtype *boxes_data = (Dtype*)boxes->mutable_data();
  auto im_info_data = (const Dtype*)im_info->data();
  Dtype zero(0);
  for (int64_t i = 0; i < boxes->valid_size(); i += 4) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), zero); //left
      boxes_data[i+1] =
          std::max(std::min(boxes_data[i+1], im_info_data[0] - 1), zero); //top
      boxes_data[i+2] =
          std::max(std::min(boxes_data[i+2], im_info_data[1] - 1), zero); // right
      boxes_data[i+3] =
          std::max(std::min(boxes_data[i+3], im_info_data[0] - 1), zero);//bottom
  }
}

template <typename Dtype, typename TargetType_H>
void filter_boxes(std::vector<int>& keep,
                  const Tensor<TargetType_H> *boxes,
                  const float min_size,
                  const Tensor<TargetType_H> *im_info) {
  const Dtype *im_info_data = (const Dtype*)im_info->data();
  const Dtype *boxes_data = (const Dtype*)boxes->data();
  Dtype im_scale = im_info_data[2];
  auto min_size_final = std::max(min_size, 1.0f);
  keep.clear();

  for (int i = 0; i < boxes->valid_size(); i += 4 ) {
      Dtype left = boxes_data[i];
      Dtype right = boxes_data[i+2];
      Dtype top = boxes_data[i+1];
      Dtype bottom = boxes_data[i+3];
      Dtype ws = right - left + 1;
      Dtype hs = bottom - top + 1;
      Dtype ws_origin_scale =
                (right - left) / im_scale + 1;
      Dtype hs_origin_scale =
                (bottom - top) / im_scale + 1;
      Dtype x_ctr = left + ws / 2;
      Dtype y_ctr = top + hs / 2;
      if (ws_origin_scale >= min_size_final && hs_origin_scale >= min_size_final &&
          x_ctr <= im_info_data[1] && y_ctr <= im_info_data[0]) {
          keep.push_back(i>>2);
      }
  }
}

template <typename Dtype>
static inline std::vector<std::pair<Dtype, int>> get_sorted_score_index(
    const std::vector<Dtype> &scores) {
    std::vector<std::pair<Dtype, int>> sorted_indices;
    sorted_indices.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        sorted_indices.emplace_back(scores[i], i);
    }

std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                   [](const std::pair<Dtype, int> &a, const std::pair<Dtype, int> &b) {
                     return a.first > b.first;
                   });
    return sorted_indices;
}

template <typename Dtype>
static inline Dtype BBoxArea(const Dtype *box, bool normalized) {
    if (box[2] < box[0] || box[3] < box[1]) {
        return static_cast<Dtype>(0.);
    } else {
        const Dtype w = box[2] - box[0];
        const Dtype h = box[3] - box[1];
        if (normalized) {
          return w * h;
        } else {
          return (w + 1) * (h + 1);
        }
    }
}

template <typename Dtype>
static inline Dtype jaccard_overlap(const Dtype *box1, const Dtype *box2, bool normalized) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
        box2[3] < box1[1]) {
        return static_cast<Dtype>(0.);
    } else {
        const Dtype inter_xmin = std::max(box1[0], box2[0]);
        const Dtype inter_ymin = std::max(box1[1], box2[1]);
        const Dtype inter_xmax = std::min(box1[2], box2[2]);
        const Dtype inter_ymax = std::min(box1[3], box2[3]);
        const Dtype inter_w = std::max(Dtype(0), inter_xmax - inter_xmin + 1);
        const Dtype inter_h = std::max(Dtype(0), inter_ymax - inter_ymin + 1);
        const Dtype inter_area = inter_w * inter_h;
        const Dtype bbox1_area = BBoxArea(box1, normalized);
        const Dtype bbox2_area = BBoxArea(box2, normalized);
        return inter_area / (bbox1_area + bbox2_area - inter_area);
    }
}

template <class Dtype, typename TargetType_H>
static inline void NMS(std::vector<int>& selected_indices,
                       Tensor<TargetType_H> *bbox,
                       std::vector<int>& indices,
                       Dtype nms_threshold,
                       float eta) {
  int64_t num_boxes = bbox->num();
  int64_t box_size = bbox->channel();

  int selected_num = 0;
  Dtype adaptive_threshold = nms_threshold;
  const Dtype *bbox_data = (const Dtype*)(bbox->data());
  selected_indices.clear();
  //while (indices.size() != 0) {
  for (int i = 0; i < indices.size(); i++) {
      //int idx = indices.back();
      auto idx = indices[i];
      bool flag = true;
      for (int kept_idx : selected_indices) {
          if (flag) {
              Dtype overlap = jaccard_overlap<Dtype>(bbox_data + idx * box_size,
                                            bbox_data + kept_idx * box_size, false);
              flag = (overlap <= adaptive_threshold);
          } else {
              break;
          }
      }
      if (flag) {
          selected_indices.push_back(idx);
          ++selected_num;
      }
      //indices.erase(indices.end() - 1);
      if (flag && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
      }
  }
}

template <typename Dtype, typename TargetType_H>
void gather(Tensor<TargetType_H>* out,
       const Tensor<TargetType_H>* in,
       std::vector<int>& index,
       const int inner_dim) {
    Shape shape = in->valid_shape();
    int index_num = index.size();
    shape[0] = index_num;
    out->reshape(shape);
    auto in_data = (const Dtype*) in->data();
    auto out_data = (Dtype*)out->data();
    for (int i = 0; i < index_num; i++) {
        memcpy(out_data + i * inner_dim, in_data + index[i] * inner_dim, sizeof(Dtype) * inner_dim);
    }
}


template <typename Dtype, typename TargetType_H>
void get_score_sorted_index(const Tensor<TargetType_H>* scores,
                            int sort_num,
                            std::vector<Dtype>& sorted_score,
                            std::vector<int>& score_index) {
   auto scores_data = (const Dtype*)scores->data();
   std::vector<std::pair<Dtype, int>> index;
   for (int i = 0; i < scores->valid_size(); i++) {
        index.emplace_back(std::make_pair(scores_data[i], i));
    }
    std::partial_sort(index.begin(), index.begin() + sort_num, index.end(),
               [](const std::pair<Dtype, int> &a, const std::pair<Dtype, int> &b) { return a.first > b.first;});

    sorted_score.resize(sort_num);
    score_index.resize(sort_num);
    for (int i = 0; i < sort_num; i++) {
        sorted_score[i] = index[i].first;
        score_index[i] = index[i].second;
    }
}

template<typename Dtype, typename TargetType_H>
void proposal_for_one_image(
      Tensor<TargetType_H> &proposals_sel,
      Tensor<TargetType_H> &scores_sel,
      Tensor<TargetType_H> &proposals,
      const Tensor<TargetType_H> &im_info_slice,//[1, 3]
      const Tensor<TargetType_H> &anchors_slice,//[H, W, A, 4]
      const Tensor<TargetType_H> &variances_slice, //[H, W, A, 4]
      const Tensor<TargetType_H> &bbox_deltas_slice,  // [1, H, W, A*4]
      const Tensor<TargetType_H> &scores_slice,       // [1, H, W, A]
      int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
      float eta) {

    int scores_num = scores_slice.valid_size();
    int index_num = 0;
    if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_num) {
        index_num = scores_num;
    } else {
        index_num = pre_nms_top_n;
    }
    std::vector<Dtype> scores_sorted;
    std::vector<int> index;
    get_score_sorted_index(&scores_slice, index_num, scores_sorted, index);

    box_coder<Dtype>(&proposals, &anchors_slice, &bbox_deltas_slice, &variances_slice, index);

    clip_tiled_boxes<Dtype>(&proposals, &im_info_slice);

    std::vector<int> keep;
    filter_boxes<Dtype>(keep, &proposals, min_size, &im_info_slice);

    if (nms_thresh <= 0) {
        gather<Dtype>(&proposals_sel, &proposals, keep, 4);
        std::vector<int> scores_index;
        for (int i = 0; i < keep.size(); i++) {
            scores_index[i] = index[keep[i]];
        }
        gather<Dtype>(&scores_sel, &scores_slice, scores_index, 1);
        return;
    }

    std::vector<int> keep_nms;
    NMS<Dtype>(keep_nms, &proposals, keep, nms_thresh, eta);

    if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.size()) {
        keep_nms.resize(post_nms_top_n);
    }

    std::vector<int> scores_index(keep_nms.size());
    for (int id = 0; id <  keep_nms.size(); id++) {
        scores_index[id] = index[keep_nms[id]];
    }
    gather<Dtype>(&scores_sel, &scores_slice, scores_index, 1);
    gather<Dtype>(&proposals_sel, &proposals, keep_nms, 4);
}

template<typename Dtype, typename TargetType_H>
void AppendProposals(Tensor<TargetType_H> *dst,
                     int64_t offset,
                     const int im_id,
                     const Tensor<TargetType_H> *src) {
  auto *out_data = (Dtype*)dst->data();
  auto *in_data = (const Dtype*)src->data();
  out_data += offset;
  for (int i = 0; i < src->valid_size()/4; i++) {
      out_data[0] = im_id;
      std::memcpy(out_data + 1, in_data, 4* sizeof(Dtype));
      out_data += 5;
      in_data += 4;
  }
}

template<typename Dtype, typename TargetType_H>
void AppendScores(Tensor<TargetType_H> *dst,
                  int64_t offset,
                  const Tensor<TargetType_H> *src) {
  auto *out_data = (Dtype*)dst->data();
  auto *in_data = (const Dtype*)src->data();
  out_data += offset;
  std::memcpy(out_data, in_data, src->valid_size() * sizeof(Dtype));
}


template <typename dtype, typename TargetType_D, typename TargetType_H>
void generate_proposals_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      GenerateProposalsParam<TargetType_D>& param) {
    auto scores = *inputs[0];
    auto bbox_deltas = *inputs[1];
    auto im_info = *inputs[2];
    auto anchors = *inputs[3];
    auto variances = *inputs[4];
    auto rpn_rois  = outputs[0];
    auto rpn_roi_probs = outputs[1];
    int pre_nms_top_n = param.pre_nms_top_n;;
    int post_nms_top_n = param.post_nms_top_n;
    float nms_thresh = param.nms_thresh;;
    float min_size = param.min_size;;
    float eta = param.eta;
    auto scores_shape = scores.valid_shape();
    auto bbox_shape = bbox_deltas.valid_shape();
    rpn_rois->reshape(Shape({bbox_deltas.valid_size() / 4, 4, 1, 1}, Layout_NCHW));
    rpn_roi_probs->reshape(Shape({scores.valid_size(), 1, 1, 1}, Layout_NCHW));
    Tensor<TargetType_H> bbox_deltas_swap;
    Tensor<TargetType_H> scores_swap;
    Tensor<TargetType_H> proposals;
    Tensor<TargetType_H> proposals_sel;
    Tensor<TargetType_H> scores_sel;

    trans<dtype>(&scores_swap, &scores);
    trans<dtype>(&bbox_deltas_swap, &bbox_deltas);

    int num_proposals = 0;
    int img_num = scores_shape[0];
    Shape im_info_slice_shape = im_info.valid_shape();
    Shape bbox_deltas_slice_shape = bbox_deltas.valid_shape();
    Shape scores_slice_shape({scores.valid_size()/ img_num, 1, 1, 1}, Layout_NCHW);
    im_info_slice_shape[0] = 1;
    bbox_deltas_slice_shape[0] = 1;
    std::vector<int> proposals_offset;
    for (int i = 0; i < img_num; i++) {
        Tensor<TargetType_H> im_info_slice((void*)((dtype*)im_info.mutable_data() + i * im_info.get_stride()[0]), TargetType_H(), 0, im_info_slice_shape);
        Tensor<TargetType_H> bbox_deltas_slice((void*)((dtype*)bbox_deltas_swap.mutable_data() + i * bbox_deltas.get_stride()[0]), TargetType_H(), 0, bbox_deltas_slice_shape);
        Tensor<TargetType_H> scores_slice((void*)((dtype*)scores_swap.mutable_data() + i * scores.get_stride()[0]), TargetType_H(), 0, scores_slice_shape);
        proposal_for_one_image<dtype>(proposals_sel,
                               scores_sel,
                               proposals,
                               im_info_slice,
                               anchors,
                               variances,
                               bbox_deltas_slice,  // [M, 4]
                               scores_slice,       // [N, 1]
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta);
      
      AppendProposals<dtype>(rpn_rois, 5 * num_proposals, i,  &proposals_sel);
      AppendScores<dtype>(rpn_roi_probs, num_proposals, &scores_sel);
      num_proposals += scores_sel.valid_size();;
      proposals_offset.push_back(num_proposals);
    }
    rpn_roi_probs->reshape(Shape({num_proposals, 1, 1, 1}, Layout_NCHW));
    rpn_rois->reshape(Shape({num_proposals, 5, 1, 1}, Layout_NCHW));
    
    std::vector<std::vector<int>> out_offset;
    out_offset.push_back(proposals_offset);
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i]->set_seq_offset(out_offset);
    }
}
template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    typedef typename DataTrait<TargetType_H, Dtype>::Dtype dtype;
    int pre_nms_top_n = 6000;
    int post_nms_top_n = 1000;
    float eta = 1.0f;
    dtype nms_thresh = 0.699999;
    dtype min_size = 0.f;
    std::string file_path = "/home/chengyujuan/baidu/sys-hic-gpu/Anakin-2.0/generate_proposals_data/";
    std::string scores_file = file_path + "result_rpn_cls_score_prob.tmp_0.txt";
    std::string bbox_deltas_file = file_path + "result_rpn_bbox_pred.tmp_1.txt";
    std::string im_info_file = file_path  + "result_im_info.txt";
    std::string anchors_file = file_path + "result_anchor_generator_0.tmp_0.txt";
    std::string variances_file = file_path + "result_anchor_generator_0.tmp_1.txt";
    TestSaberBase<TargetType_D, TargetType_H, Dtype, GenerateProposals, GenerateProposalsParam> testbase(5, 2);
    Shape bbox_deltas_shape({1, 60, 84, 84}, Layout_NCHW);
    Shape im_info_shape({1, 3, 1, 1}, Layout_NCHW);
    Shape anchors_shape({84, 84, 15, 4}, Layout_NCHW);
    Shape variances_shape({84, 84, 15, 4},  Layout_NCHW);
    Shape scores_shape({1, 15, 84, 84},Layout_NCHW);
    std::vector<Shape> input_shape_vec = {scores_shape, bbox_deltas_shape, im_info_shape, anchors_shape, variances_shape};
    GenerateProposalsParam<TargetType_D> param(pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,  eta);
    testbase.set_param(param);
    testbase.add_inputs_shape(input_shape_vec);

    Tensor<TargetType_D> scores(scores_shape);
    Tensor<TargetType_D> bbox_deltas(bbox_deltas_shape);
    Tensor<TargetType_D> im_info(im_info_shape);
    Tensor<TargetType_D> anchors(anchors_shape);
    Tensor<TargetType_D> variances(variances_shape);
    std::vector<Tensor<TargetType_D>*> input_vec;
    input_vec.push_back(&scores);
    input_vec.push_back(&bbox_deltas);
    input_vec.push_back(&im_info);
    input_vec.push_back(&anchors);
    input_vec.push_back(&variances);
    Tensor<TargetType_H> h_scores(scores_shape);
    Tensor<TargetType_H> h_bbox_deltas(bbox_deltas_shape);
    Tensor<TargetType_H> h_im_info(im_info_shape);
    Tensor<TargetType_H> h_anchors(anchors_shape);
    Tensor<TargetType_H> h_variances(variances_shape);
    
    read_tensor_from_file((dtype*)h_scores.mutable_data(), h_scores.valid_size(), scores_file.c_str());
    read_tensor_from_file((dtype*)h_bbox_deltas.mutable_data(), h_bbox_deltas.valid_size(), bbox_deltas_file.c_str());
    read_tensor_from_file((dtype*)h_im_info.mutable_data(), h_im_info.valid_size(), im_info_file.c_str());
    read_tensor_from_file((dtype*)h_anchors.mutable_data(), h_anchors.valid_size(), anchors_file.c_str());
    read_tensor_from_file((dtype*)h_variances.mutable_data(), h_variances.valid_size(), variances_file.c_str());
    scores.copy_from(h_scores);
    bbox_deltas.copy_from(h_bbox_deltas);
    im_info.copy_from(h_im_info);
    anchors.copy_from(h_anchors);
    variances.copy_from(h_variances);
    testbase.add_custom_input(input_vec);
    testbase.run_test(generate_proposals_basic<float, TargetType_D, TargetType_H>);
}

TEST(TestSaberFunc, test_func_generate_proposals) {

#ifdef USE_CUDA
    //Init the test_base
    Env<NV>::env_init();
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef AMD_GPU
    //    Env<AMD>::env_init();
    //    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
#ifdef USE_BM_PLACE
    //    Env<BM>::env_init();
    //    test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

#endif

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

